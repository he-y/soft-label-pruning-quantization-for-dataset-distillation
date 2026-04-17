# check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found"
    pip install gdown
fi

LPQLD_LABEL_DIR="./relabel_and_validate/syn_label_LPQLD"

download_dataset(){
    # Reuse LPLD distilled images and class-BN models (identical to LPQLD)
    gdown https://drive.google.com/drive/folders/18qGA3M-3pC6xpAEEFBQ3NGfTHQTz5RfH -O ./recover --folder
}

untar_dataset(){
    cd recover/syn_data_LPLD
    for f in *.tar.gz; do tar -xvf $f; done
    cd ../../
}

download_all_labels(){
    # WARNING: downloading all label variants is very large (tens of GBs).
    # This includes all IPC levels and all compression settings (x40/x80/x200 for in1k,
    # x40/x400 for in21k). Only proceed if you have sufficient disk space.
    echo "WARNING: Downloading ALL LPQLD label variants. This may require tens of GBs of disk space."
    echo "Press Ctrl+C within 10 seconds to cancel, or wait to continue..."
    sleep 10

    mkdir -p $LPQLD_LABEL_DIR
    gdown https://drive.google.com/drive/folders/1fZn0mH3_hQSh4FLzKeULAbkr9Avtys5X -O $LPQLD_LABEL_DIR --folder
}

download_recommended_labels(){
    # Download recommended labels:
    #   ImageNet-1K:  x80 (MR-50, prune_ratio=0.9) -- IPC10/20/50/100/200  (~2.8 GB total)
    #   ImageNet-21K: x400 (quantization + pruning) -- IPC10/20             (~4.3 GB total)
    mkdir -p $LPQLD_LABEL_DIR

    echo "Downloading ImageNet-1K labels (x80, MR-50, ~2.8 GB total)..."
    gdown "1q_xpjFzmA0mZP8UKDb-_EZsDeXwAjp_S" -O $LPQLD_LABEL_DIR/  # IPC10  0.07 GB
    gdown "1tpm3XD1LUYLLxAexH56mc9onlMn1XF13" -O $LPQLD_LABEL_DIR/  # IPC20  0.15 GB
    gdown "1L6d4nCfb_ROuelI0417TpSmVt7s6ga88"  -O $LPQLD_LABEL_DIR/  # IPC50  0.37 GB
    gdown "13JWSrZj_8JmbI8zPnFAgLKswRQfGI_4l"  -O $LPQLD_LABEL_DIR/  # IPC100 0.73 GB
    gdown "1HQvjvtKBUiuWXiCpfAWh7h1nyEubK2rw"  -O $LPQLD_LABEL_DIR/  # IPC200 1.47 GB

    echo "Downloading ImageNet-21K labels (x400, ~4.3 GB total)..."
    gdown "1dxCJzBLRNYlq-bxuaE42rwr1vbnMGPMG" -O $LPQLD_LABEL_DIR/  # IPC10  1.6 GB
    gdown "1Mj_RgnlkW9CFQvrUmf_DlXoGShU1-aHB" -O $LPQLD_LABEL_DIR/  # IPC20  2.7 GB
}

untar_labels(){
    cd $LPQLD_LABEL_DIR
    for f in *.tar.gz; do tar -xvf $f; done
    cd ../../
}

# Download distilled images (reused from LPLD)
download_dataset
untar_dataset

# Download LPQLD labels
# Usage: sh download.sh [true|false]
#   true  (default) -- download recommended labels only (x80 for in1k, x400 for in21k)
#   false           -- download ALL label variants (WARNING: very large, tens of GBs)
RECOMMENDED_ONLY=${1:-"true"}
if [ "$RECOMMENDED_ONLY" == "true" ]; then
    echo "Downloading recommended LPQLD labels (x80 for ImageNet-1K, x400 for ImageNet-21K)..."
    download_recommended_labels
else
    download_all_labels
fi
untar_labels
