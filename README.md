# agri-map
Some code for visualising agriculture plant spacings

## Requirements
- matplotlib
- numpy

Can be installed from in the repo with `pip install -r requirements.txt`.

## Usage
[agri_map.py](https://github.com/ES-Alexander/agri-map/agri_map.py) 
provides the classes and functionality to generate plot-spacing diagrams.
The file is also a runnable script with command-line arguments to modify 
the example output.

## Example
At the bottom of 
[agri_map.py](https://github.com/ES-Alexander/agri-map/agri_map.py)
there is an example plot spacing diagram for cocoa trees interspersed with 
temporary and permanent (TODO) shade trees. Run with `python agri_map.py` to run
as default, with results as shown
[here](https://www.notion.so/Cocoa-a020ef03d81e4ad29d1e0f310635e818).
Can also be run with command-line arguments to modify the output (run with the
`-h` flag to see the available options).
