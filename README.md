# OnShape-CAD-Parser


This is an extension of the Deepcad [project](https://github.com/ChrisWu1997/onshape-cad-parser). It's built on [Onshape-public/apikey](https://github.com/onshape-public/apikey). 

# Data
- Download ABC dataset [here](https://archive.nyu.edu/handle/2451/61215).



### Dependencies
- Clone this repo
    ```sh
    $ git clone https://gitlab.uni.lu/phd-sadil/onshape-parser.git
    $ cd onshape-public-apikey
    $ git submodule init
    $ git submodule update
    ```
##### IMPORTANT

In `apikey/onshape.py` replace
```
from urlparse import urlparse
from urlparse import parse_qs
```
with
```
from urllib.parse import urlparse
from urllib.parse import parse_qs
```

In `apikey/client.py` replace 
```
from onshape import Onshape

```
with
```
from .onshape import Onshape

```


- Install dependencies
    ```sh
    $ pip install -r requirements.txt
    ```

- Follow [this instruction](https://github.com/onshape-public/apikey/tree/master/python#running-the-app) to create a `creds.json` file in the root project folder, filled with your Onshape developer keys:
    ```json
    {
        "https://cad.onshape.com": {
            "access_key": "ACCESS KEY",
            "secret_key": "SECRET KEY"
        }
    }
    ```

### Usage
- Run on some test examples:
    ```sh
    $ python process.py --test # some test examples
    ```
    Results are saved as JSON files following the style of [Fusion360 Gallery dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/reconstruction.md).

- ABC dataset provides a large collection of Onshape CAD designs with web links [here](https://archive.nyu.edu/handle/2451/61215). To process the downloaded links in parallel, run
    ```sh
    $ python process.py --link_data_dir {path of the downloaded data}
    ```
  We collect data for our DeepCAD paper in this way.

### Note

No Note Now

### Cite

