#!/bin/bash
set -e

if [ $# -eq 0 ]
  then
    jupyter lab --config /etc/jupyter/jupyter_notebook_config.py & #&> /dev/null &
    cd /workspace/hydrodl && \
       /usr//local/lib/code-server/code-server --bind-addr 0.0.0.0:8443 \
      --user-data-dir /workspace/bkraft/code_server --extensions-dir /workspace/bkraft/code_server/extensions
  else
    exec "$@"
fi
