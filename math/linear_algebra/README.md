# Linear Algebra

This project covers various fundamental operations in linear algebra using Python, specifically focusing on matrix operations. The implementations are designed to be compliant with strict coding standards and the use of specific Python and library versions.

## Requirements

- **Python Scripts**
  - All scripts are written in Python 3.5.
  - Scripts should be compatible with numpy version 1.15.

- **Editors**
  - Allowed editors: `vi`, `vim`, `emacs`.

- **Execution Environment**
  - Scripts will be executed on Ubuntu 16.04 LTS.

- **Coding Standards**
  - Scripts must end with a new line.
  - The first line of each script should be `#!/usr/bin/env python3`.
  - Code must follow `pycodestyle` version 2.5.
  - Each module, class, and function must have appropriate documentation.

- **File Execution**
  - All files must be executable.
  - File lengths will be tested using `wc`.

## Setup

### Installing Ubuntu 16.04 and Python 3.5
Follow the instructions listed in [Using Vagrant on your personal computer](https://www.vagrantup.com/docs/installation), using `ubuntu/xenial64` as the base box.

### Verifying Python Installation
Confirm the installation of Python 3.5:
```bash
python3 -V
```

### Installing pip 19.1
```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
pip -V
```

### Installing Required Libraries
```bash
pip install --user numpy==1.15
pip install --user scipy==1.3
pip install --user pycodestyle==2.5
pip list
```

