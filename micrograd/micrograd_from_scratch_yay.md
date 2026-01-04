### Why not just use normal math?

We could have very well just used "normal" math and written `c = a * b`. But that defeats the purpose of what we are trying to build.
*   When we do just `c = a * b`, we forget that `c` came from `a` and `b`.
*   In **autograd**, we need to remember that `c` was made from `a * b`. This helps us to work backwards (**backpropagation**).


### Why are we wrapping these in a Class?

We use a class to wrap raw numbers so that we can attach extra data to them.

*   `self.data`: Holds the actual value (e.g., `2.0`).
*   **Future attributes**: Later we will be able to add:

    *   `self.grad`: To store the derivative.
    *   `self.prev`: To store the children nodes of the graph.



### The Methods

Let us now talk about the methods:

*   `__init__` (constructor): Just takes a number and stores it.
*   `__repr__`: Stands for "representation". Tells Python to print the object nicely.
    *   **Without `__repr__`**: `print(a)` would output something ugly like `<__main__.Value object at 0x7f8b1c2d3e4f>`.
    *   **With `__repr__`**: `print(a)` outputs `Value(data=2.0)`, which is much easier for us to read and debug.


### Operator Overloading (`__add__`)

The method `__add__` is what Python calls a **"Magic Method"** or **"Dunder Method"** (Double UNDERscore).

*   **What it does**: It defines what happens when you use the `+` operator on your objects.
    *   When you write `a + b`, Python internally translates this to `a.__add__(b)`.
*   **Why define it this way?**
    1.  **Unwrap**: We take the raw numbers inside (`self.data` and `other.data`) and add them.
    2.  **Wrap**: We wrap the result back into a **new `Value` object**.
        *   This is crucial! If we just returned a number, we would "fall out" of our `Value` system. By returning a `Value`, we can chain operations like `a + b + c` and keep tracking the history for autograd.


### Understanding the Internal Storage (`_children`, `_prev`)

You might be wondering about the specific variable names and structures we just added.

#### 1. Why the underscore (`_`)?
In Python, putting an underscore before a variable name (like `_children` or `_prev`) is a **convention**.
*   It signals to other programmers: **"This is internal/private. Don't touch this directly unless you know what you're doing."**
*   `self.data` is public because we want users to easily read the value.
*   `self._prev` is internal because it's part of the "plumbing" of the autograd engine.

#### 2. Why `_prev`?
This stands for **"previous"**.
*   We are building a graph. Every node (Value) needs to know where it came from.
*   If `c = a + b`, then `a` and `b` are the "previous" nodes that created `c`.
*   By storing these pointers, we can walk backwards from the result to the inputs.

#### 3. Why a `set`?
We store `_children` in a `set` (e.g., `self._prev = set(_children)`) instead of a list.
*   **Efficiency**: Sets are faster for checking if something is already there.
*   **Uniqueness**: If we do `a + a`, we don't necessarily want to store `a` twice in the same way (though for simple addition, it might not matter as much, for complex graphs, handling unique parents is cleaner). It represents the *unique* set of nodes that fed into this one.


### Visualizing the Graph (DAG)

Let's trace exactly what happens in memory when we run `d = a * b + c`.

**1. Building the Graph (Forward Pass)**
As Python executes these lines, it builds a structure of objects linked together.

*   **`a`**: Created. `data=2.0`. `_prev={}` (No parents).
*   **`b`**: Created. `data=-3.0`. `_prev={}` (No parents).
*   **`c`**: Created. `data=10.0`. `_prev={}` (No parents).
*   **`e`** (`a * b`):
    *   `data`: -6.0
    *   `_prev`: `{a, b}`  <-- **Link created!** `e` points to `a` and `b`.
    *   `_op`: `'*'`
*   **`d`** (`e + c`):
    *   `data`: 4.0
    *   `_prev`: `{e, c}`  <-- **Link created!** `d` points to `e` and `c`.
    *   `_op`: `'+'`

**2. The Resulting Structure**
This creates a **Directed Acyclic Graph (DAG)**.
*   `d` holds onto `e` and `c`.
*   `e` holds onto `a` and `b`.
*   This chain allows us to walk backwards from `d` to `a` to calculate derivatives.


### Traversing the Graph Recursively

Here's how to traverse the entire computation graph to any depth using recursion:

**Code:**
```python
# Recursive function to traverse all nodes to arbitrary depth
def trace(node, depth=0, visited=None):
    if visited is None:
        visited = set()
    
    # Avoid infinite loops in case of circular references
    if id(node) in visited:
        return
    visited.add(id(node))
    
    # Print current node with indentation based on depth
    indent = "  " * depth
    print(f"{indent}{node}")
    
    # Recursively traverse children
    for child in node._prev:
        trace(child, depth + 1, visited)

print("Complete graph traversal:")
trace(d)
```

**Output:**
```
Complete graph traversal:
Value(data=4.0)
  Value(data=10.0)
  Value(data=-6.0)
    Value(data=2.0)
    Value(data=-3.0)
```

## Understanding Recursion: Dry Run of `trace(child, depth + 1, visited)`

Let's trace through what happens when we call `trace(d)` for `d = a * b + c`:

### Initial Setup:
- `d` has `_prev = {Value(-6.0), Value(10.0)}`  (the intermediate `a*b` result and `c`)
- `Value(-6.0)` has `_prev = {Value(2.0), Value(-3.0)}`  (a and b)
- `Value(10.0)` has `_prev = {}`  (c has no children, it's a leaf)

### Execution Timeline:

**Call 1:** `trace(d, depth=0, visited={})`
- Creates new empty `visited` set
- Checks: `id(d)` not in `visited` ✓
- Adds `id(d)` to `visited` → `visited = {id(d)}`
- Prints: `"Value(data=4.0)"` (no indent, depth=0)
- Loops through `d._prev` which contains 2 nodes: `Value(-6.0)` and `Value(10.0)`
  
  **Iteration 1a:** First child is `Value(-6.0)`
  - **Calls:** `trace(Value(-6.0), depth=1, visited={id(d)})`  ← **This is the recursive call!**
  
  **Call 2:** `trace(Value(-6.0), depth=1, visited={id(d)})`
  - Checks: `id(Value(-6.0))` not in `visited` ✓
  - Adds to `visited` → `visited = {id(d), id(Value(-6.0))}`
  - Prints: `"  Value(data=-6.0)"` (2 spaces, depth=1)
  - Loops through `Value(-6.0)._prev` which contains: `Value(2.0)` and `Value(-3.0)`
    
    **Iteration 2a:** First child is `Value(2.0)`
    - **Calls:** `trace(Value(2.0), depth=2, visited={id(d), id(Value(-6.0))})`
    
    **Call 3:** `trace(Value(2.0), depth=2, visited={...})`
    - Checks: `id(Value(2.0))` not in `visited` ✓
    - Adds to `visited` → `visited = {id(d), id(Value(-6.0)), id(Value(2.0))}`
    - Prints: `"    Value(data=2.0)"` (4 spaces, depth=2)
    - Loops through `Value(2.0)._prev` which is **empty** (leaf node!)
    - No children to recurse on
    - **Returns** back to Call 2
    `````markdown
    ### Why not just use normal math?

    We could have very well just used "normal" math and written `c = a * b`. But that defeats the purpose of what we are trying to build.
    *   When we do just `c = a * b`, we forget that `c` came from `a` and `b`.
    *   In **autograd**, we need to remember that `c` was made from `a * b`. This helps us to work backwards (**backpropagation**).


    ### Why are we wrapping these in a Class?

    We use a class to wrap raw numbers so that we can attach extra data to them.

    *   `self.data`: Holds the actual value (e.g., `2.0`).
    *   **Future attributes**: Later we will be able to add:

        *   `self.grad`: To store the derivative.
        *   `self.prev`: To store the children nodes of the graph.



    ### The Methods

    Let us now talk about the methods:

    *   `__init__` (constructor): Just takes a number and stores it.
    *   `__repr__`: Stands for "representation". Tells Python to print the object nicely.
        *   **Without `__repr__`**: `print(a)` would output something ugly like `<__main__.Value object at 0x7f8b1c2d3e4f>`.
        *   **With `__repr__`**: `print(a)` outputs `Value(data=2.0)`, which is much easier for us to read and debug.



    ### Operator Overloading (`__add__`)

    The method `__add__` is what Python calls a **"Magic Method"** or **"Dunder Method"** (Double UNDERscore).

    *   **What it does**: It defines what happens when you use the `+` operator on your objects.
        *   When you write `a + b`, Python internally translates this to `a.__add__(b)`.
    *   **Why define it this way?**
        1.  **Unwrap**: We take the raw numbers inside (`self.data` and `other.data`) and add them.
        2.  **Wrap**: We wrap the result back into a **new `Value` object**.
            *   This is crucial! If we just returned a number, we would "fall out" of our `Value` system. By returning a `Value`, we can chain operations like `a + b + c` and keep tracking the history for autograd.


    ### Understanding the Internal Storage (`_children`, `_prev`)

    You might be wondering about the specific variable names and structures we just added.

    #### 1. Why the underscore (`_`)?
    In Python, putting an underscore before a variable name (like `_children` or `_prev`) is a **convention**.
    *   It signals to other programmers: **"This is internal/private. Don't touch this directly unless you know what you're doing."**
    *   `self.data` is public because we want users to easily read the value.
    *   `self._prev` is internal because it's part of the "plumbing" of the autograd engine.

    #### 2. Why `_prev`?
    This stands for **"previous"**.
    *   We are building a graph. Every node (Value) needs to know where it came from.
    *   If `c = a + b`, then `a` and `b` are the "previous" nodes that created `c`.
    *   By storing these pointers, we can walk backwards from the result to the inputs.

    #### 3. Why a `set`?
    We store `_children` in a `set` (e.g., `self._prev = set(_children)`) instead of a list.
    *   **Efficiency**: Sets are faster for checking if something is already there.
    *   **Uniqueness**: If we do `a + a`, we don't necessarily want to store `a` twice in the same way (though for simple addition, it might not matter as much, for complex graphs, handling unique parents is cleaner). It represents the *unique* set of nodes that fed into this one.


    ### Visualizing the Graph (DAG)

    Let's trace exactly what happens in memory when we run `d = a * b + c`.

    **1. Building the Graph (Forward Pass)**
    As Python executes these lines, it builds a structure of objects linked together.

    *   **`a`**: Created. `data=2.0`. `_prev={}` (No parents).
    *   **`b`**: Created. `data=-3.0`. `_prev={}` (No parents).
    *   **`c`**: Created. `data=10.0`. `_prev={}` (No parents).
    *   **`e`** (`a * b`):
        *   `data`: -6.0
        *   `_prev`: `{a, b}`  <-- **Link created!** `e` points to `a` and `b`.
        *   `_op`: `'*'`
    *   **`d`** (`e + c`):
        *   `data`: 4.0
        *   `_prev`: `{e, c}`  <-- **Link created!** `d` points to `e` and `c`.
        *   `_op`: `'+'`

    **2. The Resulting Structure**
    This creates a **Directed Acyclic Graph (DAG)**.
    *   `d` holds onto `e` and `c`.
    *   `e` holds onto `a` and `b`.
    *   This chain allows us to walk backwards from `d` to `a` to calculate derivatives.


    ### Traversing the Graph Recursively

    Here's how to traverse the entire computation graph to any depth using recursion:

    **Code:**
    ```python
    # Recursive function to traverse all nodes to arbitrary depth
    def trace(node, depth=0, visited=None):
        if visited is None:
            visited = set()
    
        # Avoid infinite loops in case of circular references
        if id(node) in visited:
            return
        visited.add(id(node))
    
        # Print current node with indentation based on depth
        indent = "  " * depth
        print(f"{indent}{node}")
    
        # Recursively traverse children
        for child in node._prev:
            trace(child, depth + 1, visited)

    print("Complete graph traversal:")
    trace(d)
    ```

    **Output:**
    ```
    Complete graph traversal:
    Value(data=4.0)
      Value(data=10.0)
      Value(data=-6.0)
        Value(data=2.0)
        Value(data=-3.0)
    ```

    ## Understanding Recursion: Dry Run of `trace(child, depth + 1, visited)`

    Let's trace through what happens when we call `trace(d)` for `d = a * b + c`:

    ### Initial Setup:
    - `d` has `_prev = {Value(-6.0), Value(10.0)}`  (the intermediate `a*b` result and `c`)
    - `Value(-6.0)` has `_prev = {Value(2.0), Value(-3.0)}`  (a and b)
    - `Value(10.0)` has `_prev = {}`  (c has no children, it's a leaf)

    ### Execution Timeline:

    **Call 1:** `trace(d, depth=0, visited={})`
    - Creates new empty `visited` set
    - Checks: `id(d)` not in `visited` ✓
    - Adds `id(d)` to `visited` → `visited = {id(d)}`
    - Prints: `"Value(data=4.0)"` (no indent, depth=0)
    - Loops through `d._prev` which contains 2 nodes: `Value(-6.0)` and `Value(10.0)`
  
      **Iteration 1a:** First child is `Value(-6.0)`
      - **Calls:** `trace(Value(-6.0), depth=1, visited={id(d)})`  ← **This is the recursive call!**
  
      **Call 2:** `trace(Value(-6.0), depth=1, visited={id(d)})`
      - Checks: `id(Value(-6.0))` not in `visited` ✓
      - Adds to `visited` → `visited = {id(d), id(Value(-6.0))}`
      - Prints: `"  Value(data=-6.0)"` (2 spaces, depth=1)
      - Loops through `Value(-6.0)._prev` which contains: `Value(2.0)` and `Value(-3.0)`
    
        **Iteration 2a:** First child is `Value(2.0)`
        - **Calls:** `trace(Value(2.0), depth=2, visited={id(d), id(Value(-6.0))})`
    
        **Call 3:** `trace(Value(2.0), depth=2, visited={...})`
        - Checks: `id(Value(2.0))` not in `visited` ✓
        - Adds to `visited` → `visited = {id(d), id(Value(-6.0)), id(Value(2.0))}`
        - Prints: `"    Value(data=2.0)"` (4 spaces, depth=2)
        - Loops through `Value(2.0)._prev` which is **empty** (leaf node!)
        - No children to recurse on
        - **Returns** back to Call 2
    
        **Iteration 2b:** Second child is `Value(-3.0)`
        - **Calls:** `trace(Value(-3.0), depth=2, visited={...})`
    
        **Call 4:** `trace(Value(-3.0), depth=2, visited={...})`
        - Checks: `id(Value(-3.0))` not in `visited` ✓
        - Adds to `visited`
        - Prints: `"    Value(data=-3.0)"` (4 spaces, depth=2)
        - No children (leaf node)
        - **Returns** back to Call 2
  
      - Call 2 finishes its loop, **Returns** back to Call 1
  
      **Iteration 1b:** Second child is `Value(10.0)` (this is `c`)
      - **Calls:** `trace(Value(10.0), depth=1, visited={...})`
  
      **Call 5:** `trace(Value(10.0), depth=1, visited={...})`
      - Checks: `id(Value(10.0))` not in `visited` ✓
      - Adds to `visited`
      - Prints: `"  Value(data=10.0)"` (2 spaces, depth=1)
      - No children (leaf node)
      - **Returns** back to Call 1

    - Call 1 finishes, **Done!**

    ### Key Points:
    1. **`depth + 1`**: Each time we go deeper into the graph, we increment depth. This creates the indentation.
    2. **`visited` parameter**: Passed to all recursive calls so everyone shares the same "already seen" tracking.
    3. **Recursion unwinds**: After processing all children at one level, the function returns to its caller and continues where it left off.


    ### Gradient Ascent

    ```python
    a.data += 0.01 * a.grad
    b.data += 0.01 * b.grad
    c.data += 0.01 * c.grad
    f.data += 0.01 * f.grad
    ```

    The reason we apply this update only to these variables is that they are the **independent variables** (leaf nodes).

    When we **ADD** a small step in the direction of the gradient, we expect $L$ to increase. This is known as **Gradient Ascent**.

    ### Backward method (notes)

    - **Purpose:** `backward()` computes $\\partial L/\\partial x$ for every `Value` node `x` reachable from the output scalar `L` by running reverse-mode automatic differentiation (backprop).

    - **High-level idea:** propagate gradients from the output to inputs using the chain rule. If `out = f(x,y,...)` then for each parent `p`:
        - `p.grad += out.grad * (\\partial out / \\partial p)`
        - `out.grad` is the incoming gradient $\\partial L/\\partial \\text{out}$ (set to `1.0` at the root).

    - **Algorithm used by `backward()`:**
        1. Build a topological ordering of nodes reachable from the output (parents before children).
        2. Zero all `.grad` values for nodes in that ordering.
        3. Set `output.grad = 1.0` (since $\\partial L/\\partial L = 1$).
        4. Traverse nodes in reverse topological order and call each node's `_backward()`, which must add (`+=`) local contributions into its parents.

    - **Why topological order?** ensures every node's `out.grad` is fully accumulated from all children before its `_backward()` runs.

    - **Local derivatives (`_backward()` math):**
        - Addition: `out = x + y` → $\\partial out/\\partial x = 1$, $\\partial out/\\partial y = 1$ → `x.grad += out.grad`, `y.grad += out.grad`.
        - Multiplication: `out = x * y` → $\\partial out/\\partial x = y$, $\\partial out/\\partial y = x$ → `x.grad += out.grad * y`, `y.grad += out.grad * x`.
        - tanh: `out = tanh(x)` → $\\partial out/\\partial x = 1-\\tanh^2(x) = 1 - out.data**2` → `x.grad += out.grad * (1 - out.data**2)`.

    - **Implementation gotchas:**
        - `_backward()` must use accumulation (`+=`), not assignment (`=`), because multiple children may contribute to the same parent.
        - Reset grads to zero before running `backward()` to avoid stale values interfering.
        - Do not call a single node's `_backward()` in isolation (e.g., `o._backward()`); that only applies a single local Jacobian and doesn't propagate through the whole graph. Use `output.backward()` to run the full reverse pass.
        - Visualizations (e.g., `draw_dot(o)`) are static snapshots — redraw after computing grads to see updated grad values.

    - **Sanity check example:** if `o = tanh(n)` and `o.grad = 1.0`, then `n.grad = 1.0 * (1 - o.data**2)`. With `o.data ≈ 0.7071`, `n.grad = 0.5`.

    - **Pseudocode for `backward()`:**

    ```
    topo = []
    visited = set()
    def build(v):
            if v not in visited:
                    visited.add(v)
                    for p in v._prev:
                            build(p)
                    topo.append(v)
    build(output)
    for v in topo:
            v.grad = 0.0
    output.grad = 1.0
    for v in reversed(topo):
            v._backward()   # each _backward must do parent.grad += local * v.grad
    ```

    This function ensures each node is visited exactly once (left-to-right) and
    appends a node to `topo` only after all nodes it depends on have been processed.

    ### Bug: gradient gets overwritten when a variable is reused (e.g. `b = a + a`)

    - **Symptom**

                    The exact form of a backward pass is up to you: you can implement tiny local gradients for simple operations (like `+` or `*`) or write composite backward passes for more complex functions (like `tanh`). What matters is that each operation supplies its local derivative, you chain those local gradients via the chain rule, and you propagate gradients from outputs back to inputs.
    - **Why this happens**
      - Using a `set` for `_prev` also collapses duplicate references, hiding multiplicity.

    - **Fix**
      2. Preserve duplicate references when storing parents (use a list instead of a set), or otherwise account for multiplicity when building gradients.
      3. Ensure `backward()` zeroes `.grad` before running and executes a reverse-topological traversal so upstream `out.grad` is complete.

    - **Code snippets:**

    ```python
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = list(_children)  # preserve multiplicity
        self.label = label
    ```

    Accumulate in `_backward()` closures:
    ```python
        out = Value(self.data + other.data, (self, other), "+")
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    ```

    Proper `backward()` pattern:
    ```python
    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        for v in topo:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    After these fixes, `a = Value(3.0); b = a + a; b.backward()` should give `a.grad == 2.0`.
    ````



    ### Wrapper and multiplication notes

    ```python
    ```

    ensures other is a Value so your Value methods can assume they get a Value (wraps raw numbers into Value).

    a*2 works, but 2*a would not

    a*2 calls a.__mul__(2) (your Value.__mul__ runs); 2*a calls 2.__mul__(a) (int's __mul__ doesn't know Value).

    hence introducing the rmul operator

    ### Division as multiplication by reciprocal

    Division is a special case of multiplication by the reciprocal:

    ```text
    a / b
    a * (1/b)
    a * (b**-1)
    ```

    So you can implement `__truediv__` by wrapping `other` as a `Value` and returning `self * (other ** -1)` (i.e., multiply by the reciprocal).

### Reset gradients between optimization steps

It's important to reset parameter gradients to zero before each `backward()` call when doing iterative optimization, otherwise gradients accumulate across steps and the effective update becomes much larger than intended.

```python
for k in range(20):

    # forward pass
    ypred = [n(x) for x in xs]

    loss = sum(((yout - ygt)**2 for yout, ygt in zip(ypred, ys)), Value(0.0))

    # backward pass: reset gradients to zero before backward
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update rule with learning rate
    for p in n.parameters():
        p.data += -0.04 * p.grad

    print(k, loss.data)
```

Don't forget to reset gradients to zero each time; otherwise they will accumulate and cause excessively large or incorrect updates. In our simple example this accidentally made convergence appear easy, but it's misleading and can lead to buggy training behaviour in general.

