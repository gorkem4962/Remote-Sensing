# Remote-Sensing-Data
This is for the milestones. In the first step please install milestone.01.tar from https://tubcloud.tu-berlin.de/s/AD48nXCfSpmPFjS and don't put the downloaded file to the repository. Don't upload to big data, if its not necessary. Its very helpful if you have installed pip, cmake. 

<p>To use our depedencies please install peotry</p>

```curl -sSL https://install.python-poetry.org | python3 -```

<p>Then go to the project folder and write</p>

```poetry install``` 

<p>If you get some error by using that command. That means some of the libaries could not be installed. Make sure that you have before installed python, clang or gcc. If you get an error for pyarrow, then please install apache-arrow with forexample homebrew or pip. Afterwards write  </p>

```pip wheel --no-cache-dir --use-pep517 "pyarrow (==17.0.0)"```

If you get an error with geospandas please install proj. 




