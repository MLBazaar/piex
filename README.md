[![PyPI Shield](https://img.shields.io/pypi/v/piex.svg)](https://pypi.python.org/pypi/piex)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/piex.svg?branch=master)](https://travis-ci.org/HDI-Project/piex)

# Pipeline Explorer

Classes and functions to explore and reproce the performance obtained by
thousands of MLBlocks pipelines and templates accross hundreds of datasets.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/piex
- Homepage: https://github.com/HDI-Project/piex


## Getting Started

### Installation

```bash
git clone git@github.com:HDI-Project/piex.git
cd piex
pip install -e .
```

### PipelineExplorer

The *PipelineExplorer* class provides methods to download the results from previous
tests executions from S3, see which pipelines obtained the best scores and load them
as a dictionary, ready to be used by an MLPipeline.

To start working with it, it needs to be given the name of the S3 Bucket from which
the data will be downloaded.

```
from piex.explorer import PipelineExplorer

piex = PipelineExplorer('ml-pipelines.2018')
```
