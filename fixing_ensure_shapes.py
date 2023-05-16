import pathlib
import json

def is_valid_node(node):
    return 'EnsureShape' not in node.get('name')

def parse_node_inputs(node, nodes_by_name):
    inputs = node.get('input', [])
    for i in range(len(inputs)):
        input = inputs[i]
        if 'EnsureShape' in input:
            ensure_shape_node = nodes_by_name[input]
            ensure_shape_node_inputs = ensure_shape_node.get('input')
            if len(ensure_shape_node_inputs) != 1:
                raise Exception(f'Invalid node named: {input}')
            inputs[i] = ensure_shape_node_inputs[0]

    return {
        **node,
        'inputs': inputs,
    }

def get_nodes_as_dict(nodes):
    nodes_by_name = {}
    for node in nodes:
        name = node.get('name')
        if name in nodes_by_name:
            raise Exception(name)
        nodes_by_name[name] = node
    return nodes_by_name

def remove_ensure_shape_nodes(model):    
    model_topology = model.get('modelTopology')
    nodes = model_topology.get('node')
    nodes_dict = get_nodes_as_dict(nodes)

    nodes = [parse_node_inputs(n, nodes_dict) for n in nodes if is_valid_node(n)]    
    
    return {
        **model,
        'modelTopology': {
            **model_topology,
            'node': nodes,
        }
    }
    
def remove_ensure_shape_nodes_from_model_json(pathname: str | pathlib.Path):
    with open(pathname, 'r') as f:
        model = json.load(f)
    with open(pathname, 'w') as f:
        f.write(json.dumps(remove_ensure_shape_nodes(model)))
