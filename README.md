# sam4onnx
A very simple tool to rewrite parameters such as attributes and constants for OPs in ONNX models. **S**imple **A**ttribute and Constant **M**odifier for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sam4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sam4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sam4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sam4onnx?color=2BAF2B)](https://pypi.org/project/sam4onnx/) [![CodeQL](https://github.com/PINTO0309/sam4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sam4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/170155850-65e2f103-baa9-4061-a268-020f0c8bc6f8.png" />
</p>

# Key concept
- [x] Specify an arbitrary OP name and Constant type INPUT name or an arbitrary OP name and Attribute name, and pass the modified constants to rewrite the parameters of the relevant OP.
- [x] Two types of input are accepted: .onnx file input and onnx.ModelProto format objects.
- [x] To design the operation to be simple, only a single OP can be specified.
- [x] Attributes and constants are forcibly rewritten, so the integrity of the entire graph is not checked in detail.


## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U sam4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```bash
$ sam4onnx -h

usage:
    sam4onnx [-h]
    --input_onnx_file_path INPUT_ONNX_FILE_PATH
    --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
    [--op_name OP_NAME]
    [--attributes NAME DTYPE VALUE]
    [--delete_attributes DELETE_ATTRIBUTES [DELETE_ATTRIBUTES ...]]
    [--input_constants NAME DTYPE VALUE]
    [--non_verbose]

optional arguments:
  -h, --help
        show this help message and exit

  --input_onnx_file_path INPUT_ONNX_FILE_PATH
        Input onnx file path.

  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        Output onnx file path.

  --op_name OP_NAME
        OP name of the attributes to be changed.
        When --attributes is specified, --op_name must always be specified.
        e.g. --op_name aaa

  --attributes NAME DTYPE VALUE
        Parameter to change the attribute of the OP specified in --op_name.
        If the OP specified in --op_name has no attributes,
        it is ignored. attributes can be specified multiple times.
        --attributes name dtype value dtype is one of
        "float32" or "float64" or "int32" or "int64" or "str".
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g.
        --attributes alpha float32 [[1.0]]
        --attributes beta float32 [1.0]
        --attributes transA int64 0
        --attributes transB int64 0

  --delete_attributes DELETE_ATTRIBUTES [DELETE_ATTRIBUTES ...]
        Parameter to delete the attribute of the OP specified in --op_name.
        If the OP specified in --op_name has no attributes,
        it is ignored. delete_attributes can be specified multiple times.
        --delete_attributes name1 name2 name3
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g. --delete_attributes alpha beta

  --input_constants NAME DTYPE VALUE
        Specifies the name of the constant to be changed.
        If you want to change only the constant,
        you do not need to specify --op_name and --attributes.
        input_constants can be specified multiple times.
        --input_constants constant_name numpy.dtype value

        e.g.
        --input_constants constant_name1 int64 0
        --input_constants constant_name2 float32 [[1.0,2.0,3.0],[4.0,5.0,6.0]]
        --input_constants constant_name3 float32 ['-Infinity']

  --non_verbose
        Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
$ python
>>> from sam4onnx import modify
>>> help(modify)

Help on function modify in module sam4onnx.onnx_attr_const_modify:

modify(
    input_onnx_file_path: Union[str, NoneType] = '',
    output_onnx_file_path: Union[str, NoneType] = '',
    onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
    op_name: Union[str, NoneType] = '',
    attributes: Union[dict, NoneType] = None,
    delete_attributes: Union[List[str], NoneType] = None,
    input_constants: Union[dict, NoneType] = None,
    non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    op_name: Optional[str]
        OP name of the attributes to be changed.
        When --attributes is specified, --op_name must always be specified.
        Default: ''
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    attributes: Optional[dict]
        Specify output attributes for the OP to be generated.
        See below for the attributes that can be specified.

        {"attr_name1": numpy.ndarray, "attr_name2": numpy.ndarray, ...}

        e.g. attributes =
            {
                "alpha": np.asarray(1.0, dtype=np.float32),
                "beta": np.asarray(1.0, dtype=np.float32),
                "transA": np.asarray(0, dtype=np.int64),
                "transB": np.asarray(0, dtype=np.int64),
            }
        Default: None
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    delete_attributes: Optional[List[str]]
        Parameter to delete the attribute of the OP specified in --op_name.
        If the OP specified in --op_name has no attributes, it is ignored.
        delete_attributes can be specified multiple times.
        --delete_attributes name1 name2 name3
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

        e.g.
        --delete_attributes alpha beta

    input_constants: Optional[dict]
        Specifies the name of the constant to be changed.
        If you want to change only the constant,
        you do not need to specify --op_name and --attributes.
        {"constant_name1": numpy.ndarray, "constant_name2": numpy.ndarray, ...}

        e.g.
        input_constants =
            {
                "constant_name1": np.asarray(0, dtype=np.int64),
                "constant_name2": np.asarray([[1.0,2.0,3.0],[4.0,5.0,6.0]], dtype=np.float32),
                "constant_name3": np.asarray([-np.inf], dtype=np.float32),
            }
        Default: None
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    modified_graph: onnx.ModelProto
        Mddified onnx ModelProto
```

## 4. CLI Execution
```bash
$ sam4onnx \
--input_onnx_file_path input.onnx \
--output_onnx_file_path output.onnx \
--op_name Transpose_17 \
--attributes perm int64 [0,1]
```

## 5. In-script Execution
```python
from sam4onnx import modify

modified_graph = modify(
    onnx_graph=graph,
    op_name="Reshape_17",
    input_constants={"241": np.asarray([1], dtype=np.int64)},
    non_verbose=True,
)
```

## 6. Sample
### 6-1. Transpose - update **`perm`**
![image](https://user-images.githubusercontent.com/33194443/163525107-f355bc2e-66d6-4a8e-bc54-2fcfc36107e8.png)
```bash
$ sam4onnx \
--input_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt.onnx \
--output_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt_mod.onnx \
--op_name Transpose_17 \
--attributes perm int64 [0,1]
```
![image](https://user-images.githubusercontent.com/33194443/163525149-64da02af-754f-40e5-916a-20f581ff0034.png)

### 6-2. Mul - update **`Constant (170)`** - From: **`2`**, To: **`1`**
![image](https://user-images.githubusercontent.com/33194443/163560084-9541140a-6368-4f4f-aced-ebdf7bf43c70.png)
```bash
$ sam4onnx \
--input_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt.onnx \
--output_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt_mod.onnx \
--op_name Mul_5 \
--input_constants 170 float32 1
```
![image](https://user-images.githubusercontent.com/33194443/163560202-15584279-58d7-4c96-b1c3-7366d165ba21.png)

### 6-3. Reshape - update **`Constant (241)`** - From: **`[-1]`**, To: **`[1]`**
![image](https://user-images.githubusercontent.com/33194443/163560715-21e0ab88-7859-4b52-adb4-c4d902525ac3.png)
```bash
$ sam4onnx \
--input_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt.onnx \
--output_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt_mod.onnx \
--op_name Reshape_34 \
--input_constants 241 int64 [1]
```
![image](https://user-images.githubusercontent.com/33194443/163561022-2e3dae84-7c6e-4ed0-9644-2248f91ab2ab.png)

## 7. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
