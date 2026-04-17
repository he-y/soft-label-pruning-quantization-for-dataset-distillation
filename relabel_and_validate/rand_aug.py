"""
Enhanced RandAugment implementation with exact parameter capture for perfect reproducibility using triplet storage format.
Dependencies: PIL, numpy, torch, matplotlib
"""
import time
import os
import random
import re
import math
import numpy as np
from functools import partial
from typing import Dict, List, Optional, Union
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from timm.data.auto_augment import _LEVEL_DENOM, _HPARAMS_DEFAULT, rand_augment_ops, rand_augment_choices
from timm.data.auto_augment import NAME_TO_OP, LEVEL_TO_ARG, _RAND_TRANSFORMS, _RAND_INCREASING_TRANSFORMS
from timm.data.auto_augment import _get_weighted_transforms
import timm.data.auto_augment as auto_augment

# Standard AugmentOp class
class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else (128, 128, 128),
            resample=Image.BICUBIC,
        )
        self.magnitude_std = self.hparams.get('magnitude_std', 0)
        self.magnitude_max = self.hparams.get('magnitude_max', None)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else -99 # NOTE: align batch
        return self.aug_fn(img, *level_args, **self.kwargs)


# Enhanced RandAugment with triplet storage format
class RandAugmentWithExactParams:
    """RandAugment that captures exact final parameters for perfect reproduction using triplet format"""
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights
        
    def __call__(self, img, saved_config=None):
        if saved_config is None:  # 'fkd_save' mode
            # Choose operations - allow randomness to progress naturally
            op_indices = np.random.choice(
                range(len(self.ops)),
                self.num_layers,
                replace=self.choice_weights is None,
                p=self.choice_weights,
            ).tolist()
            
            # Create a list of triplets: (op_index, decision, parameters)
            triplets = []
            
            # Apply operations while capturing parameters
            for op_idx in op_indices:
                op = self.ops[op_idx]

                # if isinstance(op.kwargs['resample'], (list, tuple)):
                #     chosen_resample = random.choice(op.kwargs['resample'])
                #     op_kwargs = op.kwargs.copy()
                #     op_kwargs['resample'] = chosen_resample
                # else:
                #     op_kwargs = op.kwargs
                
                # Decision to apply or not
                apply_decision = random.random()
                apply_op = op.prob >= 1.0 or apply_decision < op.prob
                
                if apply_op:
                    # Generate magnitude with randomness
                    magnitude = op.magnitude
                    if op.magnitude_std > 0:
                        if op.magnitude_std == float('inf'):
                            magnitude = random.uniform(0, magnitude)
                        else:
                            magnitude = random.gauss(magnitude, op.magnitude_std)
                    
                    # Clip magnitude
                    upper_bound = op.magnitude_max or _LEVEL_DENOM
                    magnitude = max(0., min(magnitude, upper_bound))
                    
                    # Get the exact final parameters
                    exact_params = self._get_exact_parameters(op, magnitude)
                    assert (type(exact_params) == int) or (len(exact_params) <=1), "Root of cause"
                    if op.level_fn is not None:
                        img = op.aug_fn(img, *exact_params, **op.kwargs)
                        exact_params = exact_params[0]
                    else:
                        # Apply operation without any parameters
                        img = op.aug_fn(img, **op.kwargs)
                else:
                    # Store None for operations that weren't applied
                    exact_params = -99    # to-keep the same length for the batch, NOTE: it may use additional storage
                
                # Store as a triplet
                triplets.append((op_idx, apply_decision, exact_params))
            
            return img, triplets
                
        else:  # 'fkd_load' mode
            """
            saved_config: should be a list of triplets (op_index, decision, parameters)
            """
            for triplet in saved_config:
                op_idx = triplet[0]
                decision = triplet[1]
                exact_params = triplet[2]
                
                op = self.ops[op_idx]
                apply_op = (op.prob >= 1.0) or (decision < op.prob)
                
                if apply_op:
                    # IMPORTANT: Match the exact function call pattern from original code
                    if exact_params == -99:
                        # For operations with no parameters
                        img = op.aug_fn(img, **op.kwargs)
                    else:
                        # For operations with parameters
                        if op.aug_fn == auto_augment.posterize:
                            exact_params = int(exact_params)
                        elif op.aug_fn == auto_augment.solarize:
                            exact_params = int(exact_params)
                        elif op.aug_fn == auto_augment.solarize_add:
                            exact_params = int(exact_params)
                        
                        img = op.aug_fn(img, exact_params, **op.kwargs)
            
            return img, saved_config
    
    def _get_exact_parameters(self, op, magnitude):
        """Get exact final parameters for the operation including any internal randomness"""
        # For operations with no level function, no parameters needed
        if op.level_fn is None:
            return -99  # NOTE: to align size in a batch
        
        # For operations that use _randomly_negate internally
        if op.name in ['Rotate', 'ShearX', 'ShearY', 'TranslateXRel', 'TranslateYRel']:
            # Apply the similar calculation as in the level functions, but capture the exact result
            if op.name == 'Rotate':
                # Recreate _rotate_level_to_arg
                level = (magnitude / _LEVEL_DENOM) * 30.
                # Do the random negation directly so we capture the exact value
                level = -level if random.random() > 0.5 else level
                return (level,)
            
            elif op.name in ['ShearX', 'ShearY']:
                # Recreate _shear_level_to_arg
                level = (magnitude / _LEVEL_DENOM) * 0.3
                level = -level if random.random() > 0.5 else level
                return (level,)
            
            elif op.name in ['TranslateXRel', 'TranslateYRel']:
                # Recreate _translate_rel_level_to_arg
                translate_pct = op.hparams.get('translate_pct', 0.45)
                level = (magnitude / _LEVEL_DENOM) * translate_pct
                level = -level if random.random() > 0.5 else level
                return (level,)
        
        # For all other operations, just apply the level function normally
        # and capture the result
        return op.level_fn(magnitude, op.hparams)


# Helper functions to create RandAugment transforms
def rand_augment_ops(
        magnitude=10,
        prob=0.5,
        hparams=None,
        transforms=None,
):
    """Create a list of augmentation operations."""
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
        
    return [AugmentOp(
        name, prob=prob, magnitude=magnitude, hparams=hparams) for name in transforms]


def rand_augment_transform_with_exact_params(
        config_str,
        hparams=None,
        transforms=None,
):
    """Create a RandAugment transform with exact parameter capture"""
    # Parse configuration string
    magnitude = _LEVEL_DENOM
    num_layers = 2
    increasing = False
    prob = 0.5
    hparams = hparams or _HPARAMS_DEFAULT
    
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        if c.startswith('t'):
            val = str(c[1:])
            if transforms is None:
                transforms = val
        else:
            cs = re.split(r'(\d.*)', c)
            if len(cs) < 2:
                continue
            key, val = cs[:2]
            if key == 'mstd':
                mstd = float(val)
                if mstd > 100:
                    mstd = float('inf')
                hparams.setdefault('magnitude_std', mstd)
            elif key == 'mmax':
                hparams.setdefault('magnitude_max', int(val))
            elif key == 'inc':
                if bool(val):
                    increasing = True
            elif key == 'm':
                magnitude = int(val)
            elif key == 'n':
                num_layers = int(val)
            elif key == 'p':
                prob = float(val)
            else:
                assert False, 'Unknown RandAugment config section'

    if isinstance(transforms, str):
        transforms = rand_augment_choices(transforms, increasing=increasing)
    elif transforms is None:
        transforms = _RAND_INCREASING_TRANSFORMS if increasing else _RAND_TRANSFORMS

    choice_weights = None
    if isinstance(transforms, Dict):
        transforms, choice_weights = _get_weighted_transforms(transforms)

    ra_ops = rand_augment_ops(magnitude=magnitude, prob=prob, hparams=hparams, transforms=transforms)
    return RandAugmentWithExactParams(ra_ops, num_layers, choice_weights=choice_weights)


def test_reproducibility_exact_params(img_path=None, num_trials=100, config_str='rand-m6-n3-mstd1.0'):
    """Test if augmentation is perfectly reproducible using exact parameter capture"""
    print(f"Testing RandAugment reproducibility with exact parameter capture ({num_trials} trials)")
    
    # Create or load a test image
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # Initialize RandAugment
    hparams = _HPARAMS_DEFAULT.copy()
    ra = rand_augment_transform_with_exact_params(config_str, hparams)
    
    # Run multiple trials
    successes = 0
    failures = []
    
    for trial in range(num_trials):
        # Apply augmentation and save parameters
        set_random_seed()
        img_aug1, triplets = ra(img, None)
        
        set_random_seed()
        # Apply again with saved parameters
        img_aug2, _ = ra(img, triplets)
        
        # Compare
        array1 = np.array(img_aug1)
        array2 = np.array(img_aug2)
        are_identical = np.array_equal(array1, array2)
        
        if are_identical:
            successes += 1
        else:
            failures.append((trial, np.max(np.abs(array1.astype(float) - array2.astype(float)))))
    
    # Report results
    success_rate = successes / num_trials
    print(f"Success rate: {successes}/{num_trials} = {success_rate:.2f}")
    
    if success_rate < 1.0:
        print("Failures:")
        for trial, max_diff in failures[:10]:  # Show first 10 failures
            print(f"  Trial {trial}: Max difference {max_diff}")
    else:
        print("✅ SUCCESS: All images were perfectly reproduced!")
    
    return success_rate == 1.0


def test_all_transforms_individually(num_trials=50, additional_debug=False):
    """Test each transform individually for reproducibility"""
    print("\n=== TESTING INDIVIDUAL TRANSFORMS ===")
    
    # Create a test image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # Get the list of all transforms
    transforms = _RAND_TRANSFORMS  # or _RAND_INCREASING_TRANSFORMS
    
    results = []
    for i, transform_name in enumerate(transforms):
        print(f"Testing transform {i}: {transform_name}")
        
        # Create RandAugment with only this transform
        hparams = _HPARAMS_DEFAULT.copy()
        hparams['magnitude_std'] = 1.0  # Use randomness in magnitude
        hparams['interpolation'] = Image.BICUBIC
        
        ra = rand_augment_transform_with_exact_params(
            'rand-m6-n1-mstd1.0', 
            hparams=hparams,
            transforms=[transform_name]
        )
        
        # If additional debugging is requested, show the parameters
        if additional_debug:
            print("  Debug: Testing exact parameter capture...")
            set_random_seed()
            img_aug1, triplets = ra(img, None)
            params_str = ', '.join([f"({op_idx}, {decision:.4f}, {params})" for op_idx, decision, params in triplets])
            print(f"  Triplets: [{params_str}]")
            set_random_seed()
            img_aug2, _ = ra(img, triplets)
            array1, array2 = np.array(img_aug1), np.array(img_aug2)
            print(f"  Identical: {np.array_equal(array1, array2)}")
        
        # Test with specified trials
        success = test_reproducibility_exact_params(
            num_trials=num_trials, 
            config_str='rand-m6-n1-mstd1.0'
        )
        
        results.append((transform_name, success))
        print("")
    
    # Summary
    print("\n=== TRANSFORM TEST SUMMARY ===")
    for name, success in results:
        status = "✅ Success" if success else "❌ Failure"
        print(f"{name:20s}: {status}")
    
    return results


def test_with_progressively_more_transforms():
    """Test with progressively more transforms to find if problems emerge with combinations"""
    print("\n=== TESTING WITH INCREASINGLY MORE TRANSFORMS ===")
    
    # Get the list of all transforms
    transforms = _RAND_TRANSFORMS
    
    results = []
    for num_transforms in [1, 2, 3, 5, 8, len(transforms)]:
        subset = transforms[:num_transforms]
        print(f"Testing with {num_transforms} transforms: {subset}")
        
        hparams = _HPARAMS_DEFAULT.copy()
        hparams['magnitude_std'] = 1.0  # Use randomness in magnitude
        hparams['interpolation'] = Image.BICUBIC
        
        # Use more operations per image as we add more transforms
        n = min(3, num_transforms)
        
        ra = rand_augment_transform_with_exact_params(
            f'rand-m6-n{n}-mstd1.0', 
            hparams=hparams,
            transforms=subset
        )
        
        # Test with 20 trials
        success = test_reproducibility_exact_params(
            num_trials=50, 
            config_str=f'rand-m6-n{n}-mstd1.0'
        )
        
        results.append((num_transforms, subset, success))
        print("")
    
    # Summary
    print("\n=== PROGRESSIVE TEST SUMMARY ===")
    for num, subset, success in results:
        status = "✅ Success" if success else "❌ Failure"
        print(f"{num} transforms: {status}")
    
    return results


def visualize_augmentation_and_reproduction(img_path=None, config_str='rand-m6-n5-mstd1.0'):
    """Visualize the original image, augmented image, and reproduction"""
    # Create or load a test image
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # Initialize RandAugment
    hparams = _HPARAMS_DEFAULT.copy()
    hparams['interpolation'] = Image.BICUBIC
    ra = rand_augment_transform_with_exact_params(config_str, hparams)
    
    # Apply augmentation and save parameters
    set_random_seed()
    img_aug1, triplets = ra(img, None)
    
    # Print the captured parameters for debugging
    print("Captured parameters (op_idx, decision, params):")
    for i, (op_idx, decision, params) in enumerate(triplets):
        op_name = ra.ops[op_idx].name if op_idx < len(ra.ops) else "Unknown"
        print(f"  Op {i}: {op_name} (idx {op_idx}), Decision: {decision:.4f}, Params: {params}")
    
    # Apply again with saved parameters
    set_random_seed()
    img_aug2, _ = ra(img, triplets)
    
    # Compare
    array1 = np.array(img_aug1)
    array2 = np.array(img_aug2)
    are_identical = np.array_equal(array1, array2)
    
    # Display results
    status = "✅ SUCCESS: Identical" if are_identical else "❌ FAILURE: Different"
    print(f"Reproduction status: {status}")
    
    if not are_identical:
        diff = np.abs(array1.astype(np.float32) - array2.astype(np.float32))
        print(f"Max difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")
    
    # Display images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_aug1)
    plt.title("Augmented Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(img_aug2)
    plt.title("Reproduced Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    if are_identical:
        plt.imshow(np.zeros_like(array1))
        plt.title("Difference (None)")
    else:
        plt.imshow(np.abs(array1 - array2), cmap='hot')
        plt.title("Difference")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_reproduction.png")
    plt.show()
    
    return are_identical, triplets

def set_random_seed():
    def generate_seed():
        return int.from_bytes(os.urandom(4), byteorder="big")
    seed = generate_seed()
    np.random.seed(seed)
    random.seed(seed)

def test_parameter_capture_debug():
    """Detailed test that shows exact parameter capture for each transform"""
    print("=== DETAILED PARAMETER CAPTURE TEST ===")
    
    # read an image
    img = Image.open(
        "/path/to/your/image.jpg"  # Replace with a valid image path for testing
    )
    hparams = _HPARAMS_DEFAULT.copy()
    hparams['magnitude_std'] = 1.0
    # hparams['interpolation'] = Image.BICUBIC
    
    # Test each transform
    for transform_name in _RAND_TRANSFORMS:
        print(f"\nTesting {transform_name} transform")
        
        # Create RandAugment with just this transform
        ra = rand_augment_transform_with_exact_params(
            'rand-m6-n1-mstd1.0',
            hparams=hparams,
            transforms=[transform_name]
        )
        
        # Perform multiple tests
        for i in range(10):
            # Apply the transform and capture parameters
            img_aug1, triplets = ra(img, None)

            # set random seed
            set_random_seed()
            
            # Print the captured parameters
            op_idx, decision, params = triplets[0]
            # change params to fp16
            if type(params) == float:
                params = np.float16(params)
            print(f"  Test {i+1}: Decision={decision:.4f}, Params={params}")
            
            # set random seed
            set_random_seed()

            # Apply with captured parameters
            img_aug2, _ = ra(img, triplets)
            
            # Check if identical
            array1 = np.array(img_aug1)
            array2 = np.array(img_aug2)
            identical = np.array_equal(array1, array2)
            
            if not identical:
                diff = np.abs(array1.astype(float) - array2.astype(float))
                print(f"    ❌ NOT IDENTICAL: Max diff={np.max(diff):.4f}, Mean diff={np.mean(diff):.4f}")
            else:
                print(f"    ✅ IDENTICAL")
    
    return True


# Main execution
if __name__ == "__main__":
    # Debug parameter capture in detail
    print("Running detailed parameter capture test...")
    test_parameter_capture_debug()
    
    # # Basic reproducibility test
    # print("\nRunning basic reproducibility test...")
    # test_reproducibility_exact_params(
    #     img_path="/path/to/your/image.jpg",
    #     num_trials=50
    # )

    # # Visualization
    # print("\nGenerating visualization...")
    # visualize_augmentation_and_reproduction(
    #     img_path="/path/to/your/image.jpg"
    # )
    
    # # Test all transforms individually
    # print("\nTesting each transform individually...")
    # test_all_transforms_individually(num_trials=10, additional_debug=True)
    
    # # Test with different transform combinations
    # print("\nTesting with different combinations of transforms...")
    # test_with_progressively_more_transforms()
