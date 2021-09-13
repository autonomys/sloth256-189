#!/bin/sh

HERE=`dirname $0`
cd "${HERE}"

PERL=${PERL:-perl}

for pl in *-x86_64.pl; do
    s=`basename $pl .pl`.asm
    (set -x; ${PERL} $pl masm > win64/$s)
    s=`basename $pl .pl`.s
    (set -x; ${PERL} $pl elf > elf/$s)
    (set -x; ${PERL} $pl mingw64 > coff/$s)
    (set -x; ${PERL} $pl macosx > mach-o/$s)
done

