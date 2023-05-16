#!/bin/bash
if [[ -z $1 ]]; then
  echo 'Provide a file argument'
else
  docker run -it -v $PWD:/code --runtime=nvidia --rm upscalerjs-maxim /bin/bash -c "python3 $@"
fi
