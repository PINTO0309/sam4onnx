#! /usr/bin/env python

import os
import sys
import ast
import traceback
from argparse import ArgumentParser
import numpy as np
import onnx
import onnx_graphsurgeon
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Variable, Constant
from typing import Optional, List

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

ATTRIBUTE_DTYPES_TO_NUMPY_TYPES = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'str': np.unicode_,
}

CONSTANT_DTYPES_TO_NUMPY_TYPES = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'str': np.unicode_,
    'unicode': np.unicode_,
    'bool': np.bool_,
    'complex64': np.complex64,
    'complex128': np.complex128,
}


def __search_op_constant_from_input_constant_name(
    graph: onnx_graphsurgeon.Graph,
    input_constant_name: str,
    op_name: str,
):
    """
    Parameters
    ----------
    graph: onnx_graphsurgeon.Graph
        Graphs to be explored.

    input_constant_name: str
        input_constant_name of the search target.

    op_name: str
        op_name of the search target.

    Returns
    -------
    input_constant_to_change: gs.Node
        constant found.
        If not found, return an None.

    new_constant_name: str
        A new name to be given only to the constant
        with the corresponding op_name if there are
        multiple constants with the same name in the model.
    """
    # Search for variable matching variable_name
    input_constant_to_change = None
    graph_nodes = None
    if op_name:
        graph_nodes = [graph_node for graph_node in graph.nodes if graph_node.name == op_name]
    else:
        graph_nodes = graph.nodes

    for graph_node in graph_nodes:
        for input in graph_node.inputs:
            if isinstance(input, Constant) and input.name == input_constant_name:
                input_constant_to_change = input
                break
            elif isinstance(input, Variable) and input.name == input_constant_name:
                if input.inputs[0].op == 'Constant':
                    input_constant_to_change = input.inputs[0]
                    break
        else:
            continue
        break

    # If the same input_constant_name constant exists in the model,
    # a new name is given only to the constant with the corresponding op_name.
    new_constant_name = None
    if input_constant_to_change is not None and op_name:
        num_of_const_with_the_same_name = sum(
            [
                1 for graph_node in graph.nodes \
                    for input in graph_node.inputs \
                        if input.name == input_constant_name
            ]
        )
        if num_of_const_with_the_same_name > 1:
            new_constant_name = f'{input_constant_name}_mod_{num_of_const_with_the_same_name}'

    # Return variable
    return input_constant_to_change, new_constant_name


def modify(
    input_onnx_file_path: Optional[str] = '',
    output_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    op_name: Optional[str] = '',
    attributes: Optional[dict] = None,
    delete_attributes: Optional[List[str]] = None,
    input_constants: Optional[dict] = None,
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:

    """
    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    op_name: Optional[str]
        OP name of the attributes to be changed.\n\
        When --attributes is specified, --op_name must always be specified.\n\
        Default: ''\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    attributes: Optional[dict]
        Specify output attributes for the OP to be generated.\n\
        See below for the attributes that can be specified.\n\n\
        {"attr_name1": numpy.ndarray, "attr_name2": numpy.ndarray, ...}\n\n\
        e.g. attributes = \n\
            {\n\
                "alpha": np.asarray(1.0, dtype=np.float32),\n\
                "beta": np.asarray(1.0, dtype=np.float32),\n\
                "transA": np.asarray(0, dtype=np.int64),\n\
                "transB": np.asarray(0, dtype=np.int64)\n\
            }\n\
        Default: None\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    delete_attributes: Optional[List[str]]
        Parameter to delete the attribute of the OP specified in --op_name.\n\
        If the OP specified in --op_name has no attributes, it is ignored.\n\
        delete_attributes can be specified multiple times.\n\
        --delete_attributes name1 name2 name3\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md\n\n\
        e.g.\n\
        --delete_attributes alpha beta

    input_constants: Optional[dict]
        Specifies the name of the constant to be changed. \n\
        If you want to change only the constant, \n\
        you do not need to specify --op_name and --attributes. \n\
        {"constant_name1": numpy.ndarray, "constant_name2": numpy.ndarray, ...} \n\n\
        e.g.\n\
        input_constants = \n\
            {\n\
                "constant_name1": np.asarray(0, dtype=np.int64),\n\
                "constant_name2": np.asarray([[1.0,2.0,3.0],[4.0,5.0,6.0]], dtype=np.float32)\n\
            }\n\
        Default: None\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    modified_graph: onnx.ModelProto
        Mddified onnx ModelProto
    """

    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    graph = gs.import_onnx(onnx_graph)

    # Search for OPs matching op_name
    node_subject_to_change = None
    if op_name:
        for graph_node in graph.nodes:
            if graph_node.name == op_name:
                node_subject_to_change = graph_node
                break

        if not node_subject_to_change:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'The OP specified in op_name did not exist in the graph.'
            )
            sys.exit(1)

    # Updating Attributes
    # attributes = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
    if node_subject_to_change:
        # Update
        if attributes:
            for update_attr_key, update_attr_value in attributes.items():
                found_flg = False
                for node_subject_to_change_attr_key in node_subject_to_change.attrs.keys():
                    if node_subject_to_change_attr_key == update_attr_key:
                        node_subject_to_change.attrs[node_subject_to_change_attr_key] = update_attr_value
                        found_flg = True
                        break
                if not found_flg:
                    node_subject_to_change.attrs[update_attr_key] = update_attr_value

        # Delete
        if delete_attributes:
            node_subject_to_change.attrs = \
                {attr_key: attr_value for attr_key, attr_value in node_subject_to_change.attrs.items() if attr_key not in delete_attributes}

    # Updating Constants
    """
    graph_node
    Mul_5 (Mul)
        Inputs: [
            Variable (1): (shape=[1, 3, 720, 1280], dtype=float32)
            Variable (170): (shape=None, dtype=None)
        ]
        Outputs: [
            Variable (171): (shape=None, dtype=None)
        ]

    graph_node
    Constant_4 (Constant)
        Inputs: [
        ]
        Outputs: [
            Variable (170): (shape=None, dtype=None)
        ]
    Attributes: OrderedDict([('value', Constant (): (shape=[], dtype=<class 'numpy.float32'>)
    LazyValues (shape=[], dtype=float32))])
    """
    if input_constants:
        for input_constant_name, input_constant_value in input_constants.items():
            constant, new_constant_name = __search_op_constant_from_input_constant_name(
                graph,
                input_constant_name,
                op_name,
            )
            if constant is not None:
                if hasattr(constant, "attrs"):
                    constant.attrs['value'] = gs.Constant(
                        name='' if new_constant_name is None else new_constant_name,
                        values=input_constant_value
                    )
                else:
                    if new_constant_name is None:
                        constant.values = input_constant_value
                    else:
                        if op_name:
                            new_constant = gs.Constant(
                                name=new_constant_name,
                                values=input_constant_value
                            )
                            graph_nodes = [graph_node for graph_node in graph.nodes if graph_node.name == op_name]
                            target_node = None
                            target_input_idx = -1
                            for graph_node in graph_nodes:
                                for idx, graph_node_input in enumerate(graph_node.inputs):
                                    if graph_node_input.name == input_constant_name:
                                        target_node = graph_node
                                        target_input_idx = idx
                                        break
                                else:
                                    continue
                                break
                            if target_node and target_input_idx > -1:
                                target_node.inputs[target_input_idx] = new_constant
                            else:
                                pass
                        else:
                            constant.values = input_constant_value

            else:
                if not non_verbose:
                    if op_name:
                        print(
                            f'{Color.YELLOW}WARNING:{Color.RESET} '+
                            f'op_name: {op_name}, input_constant_name: {input_constant_name} did not exist in the model.'
                        )
                    else:
                        print(
                            f'{Color.YELLOW}WARNING:{Color.RESET} '+
                            f'input_constant_name: {input_constant_name} did not exist in the model.'
                        )

    # Cleanup
    graph.cleanup().toposort()
    modified_graph = gs.export_onnx(graph)

    # Optimize
    new_model = None
    try:
        new_model = onnx.shape_inference.infer_shapes(modified_graph)
    except Exception as e:
        new_model = modified_graph
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )
            tracetxt = traceback.format_exc().splitlines()[-1]
            print(f'{Color.YELLOW}WARNING:{Color.RESET} {tracetxt}')

    # Save
    if output_onnx_file_path:
        onnx.save(new_model, f'{output_onnx_file_path}')

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    return new_model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='Output onnx file path.'
    )
    parser.add_argument(
        '--op_name',
        type=str,
        help=\
            'OP name of the attributes to be changed. \n'+
            'When --attributes is specified, --op_name must always be specified. \n'+
            'e.g. --op_name aaa'
    )
    parser.add_argument(
        '--attributes',
        nargs=3,
        action='append',
        help=\
            'Parameter to change the attribute of the OP specified in --op_name. \n'+
            'attributes can be specified multiple times. \n'+
            '--attributes name dtype value \n'+
            'dtype is one of "float32" or "float64" or "int32" or "int64" or "str". \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--attributes alpha float32 1.0 \n'+
            '--attributes beta float32 1.0 \n'+
            '--attributes transA int64 0 \n'+
            '--attributes transB int64 0'
    )
    parser.add_argument(
        '--delete_attributes',
        nargs='+',
        help=\
            'Parameter to delete the attribute of the OP specified in --op_name. \n'+
            'If the OP specified in --op_name has no attributes, it is ignored. \n'+
            'delete_attributes can be specified multiple times. \n'+
            '--delete_attributes name1 name2 name3 \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--delete_attributes alpha beta'
    )
    parser.add_argument(
        '--input_constants',
        type=str,
        nargs=3,
        action='append',
        help=\
            'Specifies the name of the constant to be changed. \n'+
            'If you want to change only the constant, \n'+
            'you do not need to specify --op_name and --attributes. \n'+
            'input_constants can be specified multiple times. \n'+
            '--input_constants constant_name numpy.dtype value \n\n'+
            'e.g.\n'+
            '--input_constants constant_name1 int64 0 \n'+
            '--input_constants constant_name2 float32 [[1.0,2.0,3.0],[4.0,5.0,6.0]]'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_onnx_file_path = args.output_onnx_file_path
    input_constants = args.input_constants
    op_name = args.op_name
    attributes = args.attributes
    delete_attributes = args.delete_attributes
    non_verbose = args.non_verbose

    # file existence check
    if not os.path.exists(input_onnx_file_path) or \
        not os.path.isfile(input_onnx_file_path) or \
        not os.path.splitext(input_onnx_file_path)[-1] == '.onnx':

        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The specified file (.onnx) does not exist. or not an onnx file. File: {input_onnx_file_path}'
        )
        sys.exit(1)

    # Load
    onnx_graph = onnx.load(input_onnx_file_path)
    graph = gs.import_onnx(onnx_graph)

    # op_name and attributes must always be specified at the same time.
    if (not op_name and attributes) or (not op_name and delete_attributes):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'op_name and attributes must always be specified at the same time.'
        )
        sys.exit(1)

    # attributes
    """
    attributes_tmp = {"name": value}
    """
    attributes_tmp = None
    if attributes:
        attributes_tmp = {}
        for attribute in attributes:
            # parse
            attr_name = attribute[0]
            attr_type = attribute[1]
            if attr_type == 'string':
                attr_type = 'str'
            if attr_type == 'str':
                attr_value = attribute[2]
            else:
                if ('-inf' in attribute[2].lower()) or ('-infinity' in attribute[2].lower()):
                    lower_attr = attribute[2].lower()
                    inf_count = lower_attr.count('-inf')
                    infinity_count = lower_attr.count('-infinity')
                    if (inf_count + infinity_count) > 1:
                        print(
                            f'{Color.RED}ERROR:{Color.RESET} '+
                            f'Values containing "inf" or "-inf" can only be 1D tensors. \n'+
                            f'e.g. [inf] or [-inf]'
                        )
                        sys.exit(1)
                    else:
                        attr_value = [-np.inf]

                elif ('inf' in attribute[2].lower()) or ('infinity' in attribute[2].lower()):
                    lower_attr = attribute[2].lower()
                    inf_count = lower_attr.count('inf')
                    infinity_count = lower_attr.count('infinity')
                    if (inf_count + infinity_count) > 1:
                        print(
                            f'{Color.RED}ERROR:{Color.RESET} '+
                            f'Values containing "inf" or "-inf" can only be 1D tensors. \n'+
                            f'e.g. [inf] or [-inf]'
                        )
                        sys.exit(1)
                    else:
                        attr_value = [np.inf]

                else:
                    attr_value = ast.literal_eval(attribute[2])

            # dtype check
            if attr_type not in ATTRIBUTE_DTYPES_TO_NUMPY_TYPES:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    f'The dtype that can be specified for attributes is one of the {ATTRIBUTE_DTYPES_TO_NUMPY_TYPES}. \n'+
                    f'dtype: {attr_type}'
                )
                sys.exit(1)

            # Conversion from python types to numpy types
            if isinstance(attr_value, list):
                attr_value = np.asarray(attr_value, dtype=ATTRIBUTE_DTYPES_TO_NUMPY_TYPES[attr_type])

            attributes_tmp[attr_name] = attr_value

    # input constants
    # input_constant = [constant_name, numpy.dtype, value]
    input_constants_tmp = None
    if input_constants:
        input_constants_tmp = {}
        for input_constant in input_constants:
            # Search for OPs corresponding to the name of input_constant
            constant, _ = __search_op_constant_from_input_constant_name(graph, input_constant[0], op_name)

            # None: Not found
            if not constant:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    f'input_constants not found. input_constant: {input_constant}'
                )
                sys.exit(1)

            # Parse
            constant_name = input_constant[0]
            constant_type = input_constant[1]
            constant_value = ast.literal_eval(input_constant[2])

            # dtype check
            if constant_type not in CONSTANT_DTYPES_TO_NUMPY_TYPES:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    f'The dtype that can be specified for constants is one of the {CONSTANT_DTYPES_TO_NUMPY_TYPES}. \n'+
                    f'dtype: {constant_type}'
                )
                sys.exit(1)

            # Conversion from python types to numpy types
            constant_value = np.asarray(constant_value, dtype=CONSTANT_DTYPES_TO_NUMPY_TYPES[constant_type])

            input_constants_tmp[constant_name] = constant_value

    # Model modify
    modified_graph = modify(
        input_onnx_file_path=None,
        output_onnx_file_path=output_onnx_file_path,
        onnx_graph=onnx_graph,
        op_name=op_name,
        attributes=attributes_tmp,
        delete_attributes=delete_attributes,
        input_constants=input_constants_tmp,
        non_verbose=non_verbose
    )


if __name__ == '__main__':
    main()
