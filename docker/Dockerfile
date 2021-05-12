# Docker image that combines NVIDIA's PyTorch and Jupyter.

# NVIDIA pytorch image
FROM nvcr.io/nvidia/pytorch:20.12-py3

# Switch user for setup.
USER root

# Customize bash prompt (user@docker:<~/current/path>).
RUN echo "PS1='\u@docker:\\w$ '" > ~/.bashrc && \
    # Remove NVIDIA created files from workspace.
    rm -rf /workspace/docker-examples && \
    rm -rf /workspace/examples && \
    rm -rf /workspace/tutorials && \
    rm -f  /workspace/README.md

# Install extra packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    zsh \
    sudo \
    unrar \
    imagemagick \
    nano \
    screen \
    ffmpeg \
    texlive-xetex dvipng texlive-fonts-recommended \
    # texlive-generic-recommended \
    cm-super \
    msttcorefonts -qq \
    atop && \
    rm -rf /var/lib/apt/lists/

RUN sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" && \
    sed -i 's/robbyrussell/agnoster/' ~/.zshrc

# Install required python modules.
RUN conda config --add channels conda-forge && \
    # conda config --set channel_priority strict && \
    conda install --quiet --yes \
    xarray \
    rasterio \
    zarr \
    dask \
    dask_labextension \
    nodejs \
    nco \
    cdo \
    esmf \
    seaborn \
    astropy \
    cartopy \
    imageio \
    ipython \
    snakeviz \
    autopep8 \
    nodejs \
    rsync \
    bottleneck \
    # required by ray.tunes hpbandster
    scipy \
    tensorboardx \
    tensorflow \
    python-ternary

#RUN conda install  --quiet --yes -c plotly plotly && \
    # jupyter labextension install @jupyterlab/plotly-extension && \
    #jupyter labextension install dask-labextension && \
    #jupyter labextension install jupyterlab_tensorboard

# Install private python modlues from gitlab.
RUN pip install git+https://oauth2:$ACCESS_TOKEN@git.bgc-jena.mpg.de/bkraft/geodatautils
RUN pip install \
    ray==1.0 \
    setproctitle \
    flake8 \
    py-spy \
    hpbandster \
    ConfigSpace \
    netcdf4

# This is required to fix a bug. After release, "ray" can be added to the the above pip install statement.
# RUN pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl

# Copy jupyter config file, this will be used by default when starting jupyter inside container.
COPY jupyter_notebook_config.py /etc/jupyter/

# Add user.
RUN useradd -rm -s $SHELL -N -g root -G sudo -u $NB_UID -p $(openssl passwd -1 pw) $NB_USER

# download the coder binary, untar it, and allow it to be executed
ENV DISABLE_TELEMETRY=true

RUN mkdir code-server && \
    cd code-server && \
    wget https://github.com/cdr/code-server/releases/download/3.2.0/code-server-3.2.0-linux-x86_64.tar.gz && \
    tar --strip-components=1 -xvzf code-server-3.2.0-linux-x86_64.tar.gz && \
    cd .. && \
    mv code-server /usr/local/lib/code-server

# Setup User Visual Studio Code Extentions
#ENV VSCODE_USER "/home/$NB_USER/.local/share/code-server/User"
#ENV VSCODE_EXTENSIONS "/home/$NB_USER/.local/share/code-server/extensions"

ENV WORKSPACE "/workspace/hydrodl/"

#copy settings to visualstudio
#RUN mkdir -p ${VSCODE_USER}
#COPY settings.json ~/hydrodl/.vscode/
# /home/$NB_USER/.local/share/code-server/User/
#RUN chown -R $NB_UID:$NB_GID /home/$NB_USER/

RUN mkdir -p /workspace/.cache/code-server/logs/
RUN chown -R $NB_UID:$NB_GID /workspace/.cache/

# Authorize SSH Host
RUN mkdir -p /etc/ssh/ && \
    chmod 0700 /etc/ssh/

# Add the keys and set permissions
COPY .ssh/id_rsa  /etc/ssh/id_rsa
COPY .ssh/id_rsa.pub  /etc/ssh/id_rsa.pub

RUN chmod 600 /etc/ssh/id_rsa && \
    chmod 600 /etc/ssh/id_rsa.pub

RUN chown -R $NB_UID:$NB_GID /etc/ssh/
RUN echo -e $"StrictHostKeyChecking no\nIdentityFile /etc/ssh/id_rsa" >> /etc/ssh/ssh_config

#RUN echo 'fs.inotify.max_user_watches=524288' >> /etc/sysctl.conf && \
#    sysctl -p

# Install powerline font
RUN git clone https://github.com/abertsch/Menlo-for-Powerline.git && \
    mkdir -p ~/.local/share/fonts && \
    cp Menlo-for-Powerline/Menlo* ~/.local/share/fonts && \
    rm -r Menlo-for-Powerline && \
    fc-cache -f -v

# Needed to run jupyter
RUN mkdir -p /workspace/.local/share/jupyter && \
    chmod 777 /workspace/.local/share/jupyter

# Needed by cartopy to download maps.
RUN chmod -R 777 /workspace/.local/share/

# RUN chmod -R 777 /workspace/.config

# Conda permissions.
RUN chmod -R 777 /opt/conda

# Create entrypoint that runs jupyter and code-server.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

#Switch to user.
USER $NB_UID

# Set entrypoint.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
