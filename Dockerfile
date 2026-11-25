FROM rocker/shiny-verse:4.4.3

RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=5 \
    -o Acquire::http::Timeout=60 \
    -o Acquire::https::Timeout=60 \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    cmake \
    git \
    libgit2-dev \
    libnode-dev \
    build-essential \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c( \
    'shiny', \
    'shinyWidgets', \
    'shinythemes', \
    'DT', \
    'shapviz', \
    'kernelshap', \
    'mlr3verse', \
    'mlr3', \
    'mlr3learners', \
    'mlr3tuningspaces', \
    'xgboost', \
    'data.table', \
    'ggplot2', \
    'jsonlite', \
    'pak', \
    'remotes' \
    ), repos=c('https://posit-dev.r-universe.dev', 'https://tidyverse.r-universe.dev', 'https://cloud.r-project.org'), Ncpus = 4)"

RUN R -e "install.packages( \
    c('jsonvalidate', 'logger'), \
    repos = c('https://cloud.r-project.org'), \
    type = 'source', \
    dependencies = c('Depends', 'Imports', 'LinkingTo'), \
    Ncpus = 4); \
    stopifnot(requireNamespace('jsonvalidate', quietly = TRUE)); \
    stopifnot(requireNamespace('logger', quietly = TRUE))"

RUN R -e "install.packages( \
    c('httr2', 'coro', 'later', 'promises', 'S7', 'cli', 'glue', 'lifecycle', 'R6', 'rlang', 'tibble', 'vctrs'), \
    repos = c('https://cloud.r-project.org'), \
    type = 'source', \
    dependencies = c('Depends', 'Imports', 'LinkingTo'), \
    Ncpus = 4); \
    stopifnot(packageVersion('httr2') >= '1.2.1'); \
    stopifnot(packageVersion('coro') >= '1.1.0'); \
    stopifnot(packageVersion('later') >= '1.4.0'); \
    stopifnot(packageVersion('promises') >= '1.3.1'); \
    stopifnot(packageVersion('S7') >= '0.2.0')"

RUN R -e "install.packages( \
    'ellmer', \
    repos = c('https://cloud.r-project.org'), \
    type = 'source', \
    dependencies = FALSE, \
    Ncpus = 4); \
    stopifnot(requireNamespace('ellmer', quietly = TRUE))"

RUN R -e "install.packages( \
    c('survdistr', 'cmprsk', 'riskRegression', 'remotes'), \
    repos = c('https://cloud.r-project.org'), \
    type = 'source', \
    dependencies = c('Depends', 'Imports', 'LinkingTo'), \
    Ncpus = 4); \
    remotes::install_github( \
    c('xoopR/ooplah', 'xoopR/dictionar6', 'xoopR/set6', 'xoopR/param6', 'xoopR/distr6', 'mlr-org/mlr3proba', 'mlr-org/mlr3cmprsk', 'mlr-org/mlr3extralearners'), \
    upgrade = 'never', \
    dependencies = c('Depends', 'Imports', 'LinkingTo')); \
    stopifnot(requireNamespace('mlr3proba', quietly = TRUE)); \
    stopifnot(requireNamespace('mlr3cmprsk', quietly = TRUE)); \
    stopifnot(requireNamespace('survdistr', quietly = TRUE)); \
    stopifnot(requireNamespace('mlr3extralearners', quietly = TRUE))"

RUN R -e "remotes::install_url( \
    'https://github.com/catboost/catboost/releases/download/v1.2.5/catboost-R-linux-x86_64-1.2.5.tgz', \
    INSTALL_opts = c('--no-multiarch', '--no-test-load'))"

RUN R -e "remotes::install_version( \
    'mlr3', \
    version = '1.3.0', \
    repos = 'https://cloud.r-project.org', \
    upgrade = 'never'); \
    stopifnot(as.character(packageVersion('mlr3')) == '1.3.0')"

RUN R -e "stopifnot(requireNamespace('mlr3', quietly = TRUE)); stopifnot(as.character(packageVersion('mlr3')) == '1.3.0'); stopifnot(requireNamespace('mlr3extralearners', quietly = TRUE)); stopifnot(requireNamespace('catboost', quietly = TRUE)); stopifnot(requireNamespace('ellmer', quietly = TRUE)); stopifnot(packageVersion('httr2') >= '1.2.1')"

RUN useradd -m -u 1000 hfuser

WORKDIR /home/hfuser/app

COPY --chown=hfuser:hfuser . /home/hfuser/app/

ENV MODEL_DIR=/home/hfuser/app/models

USER hfuser

EXPOSE 7860

CMD ["R", "-e", "shiny::runApp('/home/hfuser/app', host='0.0.0.0', port=7860)"]
