#!/bin/bash
test_dir = /home/nas4/user/kjh4/spear/PercepNet/bin/tests
input_wavfilepath = 
input_pcmfilepath = 
output_pcmfilepath =
sox ${input_wavfilepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${input_pcmfilepath}
./percepNet_tst ${input_pcmfilepath} percepnet_output.pcm