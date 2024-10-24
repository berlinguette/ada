import os
from hyperspec import HyperPos

_elements = ['ag_la', 'cu', 'cl', 's']

dst = ["/path/to/your/bcf_files/"]


for _dst in dst:
    # Get a list of the file paths in the XRF directory
    file_paths = [os.path.join(_dst, file) for file in os.listdir(_dst)]
    # bcf conversion to png files
    for _file in file_paths:
        if _file.endswith('.bcf'):
            file_path = _file
            h = HyperPos(path=file_path)
            for element in _elements:
                try:
                    h.make_element_map(element)
                except:
                    pass