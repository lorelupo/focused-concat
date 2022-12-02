#!/usr/bin/env bash
# Previous to this script, you should already have run
# sh/preprocess_iwslt16_with_wmt14_en_fr.sh

# Move to ./data directory
cd data

# Set vars
src=en
tgt=fr
prep=iwslt16.dnmt.fr-en
outdir=wmt14
tmp=$prep/$outdir/tmp
bpe_out=$prep/$outdir

pretrained=../checkpoints/en2fr_wmt14_transformer

HEADS=../scripts/retrieve_doc_heads.py
BPEROOT=subword-nmt/subword_nmt
BPE_CODE=$pretrained/bpecodes

# Applying BPE
rm -rf $bpe_out
rm -rf data-bin/$bpe_out
mkdir -p $bpe_out

for L in $src $tgt; do
    for f in train.$L valid.$L; do
        echo "Apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/standard/tmp/$f > $bpe_out/$f
    done
done

for L in $src $tgt; do
    f=xl_test.$L
    echo "Apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/xl_test/tmp/$f > $bpe_out/$f
done

# Build vocabularies and binarize training data
echo "Building vocabulary and binarizing data..."
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --trainpref $bpe_out/train \
    --validpref $bpe_out/valid \
    --testpref $bpe_out/xl_test \
    --srcdict $pretrained/dict.$src.txt \
    --joined-dictionary \
    --destdir data-bin/$bpe_out \
    --workers 8

cp data-bin/$prep/standard/*.heads data-bin/$bpe_out/
