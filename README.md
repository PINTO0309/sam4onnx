# [WIP] sam4onnx
A very simple tool to rewrite parameters such as attributes and constants for OPs in ONNX models. **S**imple **A**ttribute and Constant **M**odifier for **ONNX**.

# Key concept
- [ ] Specify an arbitrary OP name and Constant type INPUT name or an arbitrary OP name and Attribute name, and pass the modified constants to rewrite the parameters of the relevant OP.
- [ ] Two types of input are accepted: .onnx file input and onnx.ModelProto format objects.
- [ ] To design the operation to be simple, only a single OP can be specified.
