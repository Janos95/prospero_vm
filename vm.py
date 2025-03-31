import numpy as np

with open('parts/prospero.vm') as f:
    text = f.read().strip()

image_size = 1024
space = np.linspace(-1, 1, image_size)
(x, y) = np.meshgrid(space, -space)
v = {}

for line in text.split('\n'):
    # skip comments and empty lines
    if line.startswith('#') or not line.strip():
        continue
    [out, op, *args] = line.split()
    match op:
        case "var-x": v[out] = x
        case "var-y": v[out] = y
        case "const": v[out] = float(args[0])
        case "add": v[out] = v[args[0]] + v[args[1]]
        case "sub": v[out] = v[args[0]] - v[args[1]]
        case "mul": v[out] = v[args[0]] * v[args[1]]
        case "max": v[out] = np.maximum(v[args[0]], v[args[1]])
        case "min": v[out] = np.minimum(v[args[0]], v[args[1]])
        case "neg": v[out] = -v[args[0]]
        case "square": v[out] = v[args[0]] * v[args[0]]
        case "sqrt": v[out] = np.sqrt(v[args[0]])
        case _: raise Exception(f"unknown opcode '{op}'")
out = v[out]

with open('out.ppm', 'wb') as f: # write the image out
    f.write(f'P5\n{image_size} {image_size}\n255\n'.encode())
    f.write(((out < 0) * 255).astype(np.uint8).tobytes())