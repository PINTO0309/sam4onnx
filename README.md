# [WIP] sam4onnx
A very simple tool to rewrite parameters such as attributes and constants for OPs in ONNX models. **S**imple **A**ttribute and Constant **M**odifier for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

# Key concept
- [ ] Specify an arbitrary OP name and Constant type INPUT name or an arbitrary OP name and Attribute name, and pass the modified constants to rewrite the parameters of the relevant OP.
- [ ] Two types of input are accepted: .onnx file input and onnx.ModelProto format objects.
- [ ] To design the operation to be simple, only a single OP can be specified.

## 6. Sample
### 6-1. Transpose - update **`perm`**
![image](https://user-images.githubusercontent.com/33194443/163525107-f355bc2e-66d6-4a8e-bc54-2fcfc36107e8.png)
```bash
$ sam4onnx \
--op_name Transpose_17
--input_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt.onnx
--output_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt_mod.onnx
--attributes perm int64 [0,1]
```
![image](https://user-images.githubusercontent.com/33194443/163525149-64da02af-754f-40e5-916a-20f581ff0034.png)

### 6-2. Mul - update **`Constant (170)`** - From: **`2`**, To: **`1`**
![image](https://user-images.githubusercontent.com/33194443/163560084-9541140a-6368-4f4f-aced-ebdf7bf43c70.png)
```bash
$ sam4onnx \
--input_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt.onnx \
--output_onnx_file_path hitnet_sf_finalpass_720x1280_nonopt_mod.onnx \
--input_constants 170 float32 1
```
![image](https://user-images.githubusercontent.com/33194443/163560202-15584279-58d7-4c96-b1c3-7366d165ba21.png)
