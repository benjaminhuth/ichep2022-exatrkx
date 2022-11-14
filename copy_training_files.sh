#!/bin/bash

function copy10 {
    DEST_DIR=$(cd $1 && pwd)
    SEARCH_DIR=$(cd $2 && pwd)

    cd $SEARCH_DIR

    # "000*" to use e.g. 10 events
    find \
        -type f \
        -name "0000" \
        -exec mkdir -p $DEST_DIR/{} \; \
        -exec cp -f {} $DEST_DIR/{} \;

    #mv $DEST_DIR/tmp $DEST_DIR/tmp_10_events
}

(copy10 smeared_training ~/exatrkx/traintrack_configs/ODD-1K-smear)
(copy10 truth_training ~/exatrkx/traintrack_configs/ODD-1K-truth)

