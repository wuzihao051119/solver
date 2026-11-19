n = 20
num_elems = n * n
x = [-1 + 2 * i / n for i in range(n + 1)]
y = [-1 + 2 * i / n for i in range(n + 1)]
num_nodes = len(x) * len(y)

with open(f"assets/poisson2d{n}x{n}.msh", "w") as f:
    f.write(
        "$MeshFormat\n"
        "4.1 0 8\n"
        "$EndMeshFormat\n"
        "$Nodes\n"
    )

    f.write(f"1 {num_nodes} 1 {num_nodes}\n")
    f.write(f"2 1 0 {num_nodes}\n")

    for i in range(0, num_nodes):
        f.write(f"{i + 1}\n")

    for y_val in y:
        for x_val in x:
            f.write(f"{x_val:.1f} {y_val:.1f} 0.0\n")

    f.write(
        "$EndNodes\n"
        "$Elements\n"
    )

    f.writelines([
        f"1 {num_elems} 1 {num_elems}\n",
        f"2 1 3 {num_elems}\n",
    ])

    for i in range(n):
        for j in range(n):
            idx = i * (n + 1) + j + 1
            f.write(f"{i * n + j + 1} {idx} {idx + 1} {idx + n + 2} {idx + n + 1}\n")

    f.write("$EndElements\n")
