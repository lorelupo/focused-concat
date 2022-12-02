#!/usr/bin/env bash

# TODO(lo) doc heads could be a unique file for src and tgt language

# Move to ./data directory
cd data

# Setting variables
src=fr
tgt=en
lang=fr-en
orig=iwslt16.original.fr-en
prep=iwslt16.dnmt.$src-$tgt
outdir=standard
tmp=$prep/$outdir/tmp
mkdir -p $tmp $prep

GZ=fr-en.tgz
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt

HEADS=../scripts/retrieve_doc_heads.py

# Setting utils ################################################################

echo 'Looking for Moses github repository (for tokenization scripts)...'
DIR="./mosesdecoder"
if [ -d "$DIR" ]; then
  echo "Moses repo was already cloned here."
else
  echo 'Cloning Moses github repository.'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi


echo 'Looking for Subword NMT repository (for BPE pre-processing)...'
DIR="./subword-nmt"
if [ -d "$DIR" ]; then
  echo "Subword NMT repo was already cloned here."
else
  echo 'Cloning Subword NMT repository.'
  git clone https://github.com/rsennrich/subword-nmt.git
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

# Downloading data #############################################################

URL="https://wit3.fbk.eu/archive/2016-01/texts/fr/en/fr-en.tgz"


if [ -d "$orig" ]; then
  echo "Data are already available in local dir."
else
  mkdir -p $orig
  echo "Downloading data from ${URL}..."
  cd $orig
  wget "$URL"
  if [ -f $GZ ]; then
    echo "Data successfully downloaded."
  else
    echo "Data not successfully downloaded."
    exit
  fi
  tar zxvf $GZ
  cd ..
fi

# Preprocessing ################################################################

echo "Pre-processing train data..."
for l in $src $tgt; do
    # Select training files
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l
    # Remove lines containing url, talkid and keywords (grep -v).
    # Remove special tokens with sed -e.
    # Then tokenize (insert spaces between words and punctuation) with moses.
    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '</translator>' | \
    grep -v '</reviewer>' | \
    grep -v '</speaker>' | \
    grep -v '</keywords>' | \
    sed -e 's/<talkid>.*<\/talkid>//g' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

# Clean training files from long sentences (100 sentences longer then 175 tok),
# [not empty sentences] and sentences that highly mismatch in length (ratio)
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 0 175

echo "Lowercase everything"
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "Pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT16.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    cat $o | \
    sed '/<doc \s*/i <seg id="0">' | \
    grep '<seg id' | \
    sed -e 's/<seg id="[0-9]*">\s*//g' | \
    sed -e 's/\s*<\/seg>\s*//g' | \
    sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done

# Splitting dataset ############################################################

echo "creating train, valid, test data"
for l in $src $tgt; do
    # textual datasets
    cat $tmp/train.tags.fr-en.$l          > $tmp/train.$l
    cat $tmp/IWSLT16.TED.dev2010.fr-en.$l > $tmp/valid.$l  
    cat $tmp/IWSLT16.TED.tst2010.fr-en.$l > $tmp/test.$l
    # retrieve indices of headlines
    python $HEADS $tmp/train.$l
    python $HEADS $tmp/valid.$l
    python $HEADS $tmp/test.$l
    mv $tmp/train.$l.heads $prep/$outdir/train.$lang.$l.heads
    mv $tmp/valid.$l.heads $prep/$outdir/valid.$lang.$l.heads
    mv $tmp/test.$l.heads $prep/$outdir/test.$lang.$l.heads
done

# Print stats before BPE #######################################################

for l in $src $tgt; do
    echo "Stats for train.$l (tokenized and cleaned):"
    words=$(wc -w $tmp/train.$l | awk '{print $1;}')
    sents=$(wc -l $tmp/train.$l | awk '{print $1;}')
    printf "%10d words \n" $words
    printf "%10d sentences \n" $sents
    printf "%10s wps \n" $(echo "scale=2 ; $words / $sents" | bc)
    echo
done

# Learning and applying BPE ###################################################

BPE_TOKENS=32000
BPE_CODE=$prep/$outdir/code

# Concatenate source and target language data in TRAIN file
TRAIN=$tmp/train.en-fr
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "Learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$outdir/$f
    done
done

# Build vocabularies and binarize training data ###############################

echo "Building vocabulary and binarizing data..." 
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --trainpref $prep/$outdir/train \
    --validpref $prep/$outdir/valid \
    --testpref $prep/$outdir/test \
    --joined-dictionary \
    --destdir data-bin/$prep/$outdir

cp $prep/$outdir/*.heads data-bin/$prep/$outdir/