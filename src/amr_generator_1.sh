#!/bin/bash

JAMR_PATH=$1
FILE_PATH=$2
OUT_PATH=$3


cd ${JAMR_PATH}

. scripts/config.sh

. scripts/PARSE.sh < ${FILE_PATH} > ${OUT_PATH} 2> out.out