## Loading Names from File

```python
words=open("names.txt","r").read().splitlines()
```

This line does three operations in sequence:
1. **`open("names.txt","r")`** - Opens the file `names.txt` in read mode
2. **`.read()`** - Reads the entire file content as a single string
3. **`.splitlines()`** - Splits the string into a list of lines (removes newline characters `\n`)

Result: `words` becomes a list where each element is one name from the file.

---

## Building a Bigram Dictionary

```python
b={}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram=(ch1,ch2)
        b[bigram]=b.get(bigram,0)+1
        print(ch1, ch2)
        pass
```

This code counts **bigrams** (consecutive character pairs) in the first 3 names:

1. **`b={}`** - Initialize empty dictionary to store bigram counts

2. **`for w in words[:3]:`** - Loop through first 3 names only

3. **`chs = ["<S>"] + list(w) + ["<E>"]`** - Prepare character sequence:
   - `list(w)` converts name string into list of characters
   - Add `<S>` (start token) at beginning
   - Add `<E>` (end token) at end
   - Example: `"emma"` → `["<S>", "e", "m", "m", "a", "<E>"]`

4. **`for ch1, ch2 in zip(chs, chs[1:]):`** - Iterate over consecutive pairs:
   - `chs[1:]` shifts list by one position
   - `zip()` pairs them together: `(<S>,e)`, `(e,m)`, `(m,m)`, `(m,a)`, `(a,<E>)`

5. **`bigram=(ch1,ch2)`** - Store pair as tuple

6. **`b[bigram]=b.get(bigram,0)+1`** - Count occurrences:
   - `b.get(bigram,0)` returns current count (or 0 if not seen)
   - Add 1 to increment the count
   - Store back in dictionary

**Result**: `b` contains how many times each character follows another across all the names. This is the foundation for a character-level language model that learns which character transitions are common.

---

## Sorting Bigrams by Frequency

```python
sorted(b.items(), key=lambda kv:kv[1], reverse=True)
```

This is basically equivalent to:

```python
def extract_count(kv):
    return kv[1]

sorted(b.items(), key=extract_count, reverse=True)
```

### How sorted() processes each item:

```python
# sorted() internally calls the lambda for EACH item:

item1 = (('e', 'm'), 3)
result1 = lambda kv: kv[1]   → (lambda (('e', 'm'), 3): 3)  → returns 3

item2 = (('m', 'a'), 7)
result2 = lambda kv: kv[1]   → (lambda (('m', 'a'), 7): 7)  → returns 7

item3 = (('a', 'n'), 2)
result3 = lambda kv: kv[1]   → (lambda (('a', 'n'), 2): 2)  → returns 2

# Then sorted() compares: 3, 7, 2 → sorts by these numbers
```

**Result:** Bigrams ordered from most frequent to least frequent, making it easy to see which character transitions are most common in the dataset.

---

## Creating a 2D Bigram Count Array

```python
N = torch.zeros((27,27), dtype=torch.int32)
```

We create a **27×27 matrix** because:
- **26 letters** in the alphabet (a-z)
- **1 special character**: `.` (serves as both start and end token)
- Total: 27 characters

The matrix `N` is a 2-dimensional array where the **intersection N[i, j]** represents the **count of how many times character `i` is followed by character `j`**.

**Structure:**
- **Rows** represent the **first character** (current character)
- **Columns** represent the **second character** (next character)
- **Value at N[i, j]** = number of times character `i` → character `j` transition occurs

**Example:**  
- If 'e' is at index 5 and 'm' is at index 13
- N[5, 13] would store the count of bigram `('e', 'm')`
- If we see "emma", "emily", "emma" → N[5, 13] might be 3

This 2D representation makes it easy to:
- Look up transition probabilities
- Visualize patterns as a heatmap
- Perform matrix operations for sampling next characters

---

## Creating Character-to-Index Mappings

```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}  # Start from 1 instead of 0
stoi['.'] = 0  # Reserve index 0 for the special token
itos = {i:s for s, i in stoi.items()}
```

We need to convert characters to integers (indices) to use them in the matrix.

**Step-by-step:**

1. **`''.join(words)`** - Concatenate all names into one long string
   - Example: `['emma', 'olivia', 'ava']` → `'emmaoliviaava'`

2. **`set(...)`** - Extract unique characters
   - `'emmaoliviaava'` → `{'e', 'm', 'a', 'o', 'l', 'i', 'v'}`

3. **`sorted(list(...))`** - Convert to sorted list
   - `{'e', 'm', 'a', ...}` → `['a', 'e', 'i', 'l', 'm', 'o', 'v']`

4. **`{s:i+1 for i, s in enumerate(chars)}`** - Create string-to-index dictionary, starting from 1
   - `enumerate(chars)` gives: `(0, 'a'), (1, 'b'), (2, 'c'), ...`
   - With `i+1`: `'a':1, 'b':2, 'c':3, ..., 'z':26`

5. **`stoi['.'] = 0`** - Reserve index 0 for the special token '.'
   - `.` is used as both start and end of name marker

6. **`itos = {i:s for s, i in stoi.items()}`** - Create reverse mapping (index-to-string)
   - `{0:'.', 1:'a', 2:'b', ..., 26:'z'}`

**Final mappings:**
- `stoi` (string to index): `{'.': 0, 'a': 1, 'b': 2, ..., 'z': 26}`
- `itos` (index to string): `{0: '.', 1: 'a', 2: 'b', ..., 26: 'z'}`

---

## Populating the Bigram Matrix

```python
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

This loop fills the N matrix with actual bigram counts from all names.

**Dry run with name "emma":**

```python
w = "emma"
chs = ["."] + list("emma") + ["."]
# chs = ['.', 'e', 'm', 'm', 'a', '.']

# Iteration 1: ch1='.', ch2='e'
ix1 = stoi['.'] = 0
ix2 = stoi['e'] = 5
N[0, 5] += 1  # Count: '.' → 'e' transition

# Iteration 2: ch1='e', ch2='m'
ix1 = stoi['e'] = 5
ix2 = stoi['m'] = 13
N[5, 13] += 1  # Count: 'e' → 'm' transition

# Iteration 3: ch1='m', ch2='m'
ix1 = stoi['m'] = 13
ix2 = stoi['m'] = 13
N[13, 13] += 1  # Count: 'm' → 'm' transition

# Iteration 4: ch1='m', ch2='a'
ix1 = stoi['m'] = 13
ix2 = stoi['a'] = 1
N[13, 1] += 1  # Count: 'm' → 'a' transition

# Iteration 5: ch1='a', ch2='.'
ix1 = stoi['a'] = 1
ix2 = stoi['.'] = 0
N[1, 0] += 1  # Count: 'a' → '.' transition (end)
```

After processing all 32,033 names, N contains the complete bigram statistics.

---

## Visualizing the Bigram Matrix

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
```

This creates a **heatmap visualization** of the bigram count matrix:

1. **`plt.imshow(N, cmap='Blues')`** - Display matrix as image
   - Darker blue = higher count (more frequent bigram)
   - Lighter blue = lower count (rare bigram)

2. **Double nested loop** - For each cell (i, j) in the 27×27 grid:
   - `chstr = itos[i] + itos[j]` - Create bigram string (e.g., "ab", "em")
   - First `plt.text()` - Display bigram characters at bottom of cell
   - Second `plt.text()` - Display count value at top of cell
   - `.item()` - Convert PyTorch tensor to Python number

3. **`plt.axis('off')`** - Hide axis labels for cleaner visualization

**What you see:**
- Row 0 (starting with '.'): shows which letters commonly start names
- Column 0 (ending with '.'): shows which letters commonly end names
- Diagonal elements: doubled letters (aa, bb, cc, etc.)
- Dark patches: common transitions (like 'a' → 'n', 'e' → 'r', etc.)

---

## Converting Counts to Probabilities

```python
p = N[0].float()  # Get first row - transitions from '.'
p = p / p.sum()    # Normalize to probabilities
```

To **sample** the next character, we need probabilities, not raw counts.

**Example with simplified numbers:**

```python
# First row of N (transitions from '.' start token):
N[0] = tensor([0, 4410, 1306, 1542, ...])
         #     .    a     b     c

# Step 1: Convert to float
p = N[0].float()
# p = tensor([0., 4410., 1306., 1542., ...])

# Step 2: Compute sum
total = p.sum() = 32033.0  # Total names in dataset

# Step 3: Normalize (divide each by total)
p = p / total
# p = tensor([0.0000, 0.1377, 0.0408, 0.0481, ...])

# Verification:
p.sum() = 1.0000  # Always sums to 1 (100%)
```

**Interpretation:**
- `p[0] = 0.0000` → 0% of names start with '.'
- `p[1] = 0.1377` → 13.77% of names start with 'a'
- `p[2] = 0.0408` → 4.08% of names start with 'b'
- `p[3] = 0.0481` → 4.81% of names start with 'c'

Now we can **sample** from this distribution to pick a likely first character!

---

## Random Number Generator with Seed

```python
g = torch.Generator().manual_seed(2147483647)
```

**Purpose:** Create a generator for **reproducible** random numbers.

**Without seed (non-reproducible):**
```python
torch.rand(3)  # [0.2312, 0.8945, 0.1234]
torch.rand(3)  # [0.5671, 0.3298, 0.9012]  ← Different every time!
```

**With seed (reproducible):**
```python
g = torch.Generator().manual_seed(42)
torch.rand(3, generator=g)  # [0.8823, 0.9150, 0.3829]

# Restart and use same seed:
g = torch.Generator().manual_seed(42)
torch.rand(3, generator=g)  # [0.8823, 0.9150, 0.3829]  ← Same results!
```

**Why use seeds?**
- **Reproducibility** - Get same generated names every time
- **Debugging** - Easier to track down issues
- **Sharing** - Others can reproduce your exact results

**Generator maintains state:**
```python
g = torch.Generator().manual_seed(42)
torch.rand(3, generator=g)  # [0.8823, 0.9150, 0.3829]
torch.rand(3, generator=g)  # [0.9593, 0.3904, 0.6009]  ← Continues sequence

# Reset to start over:
g.manual_seed(42)
torch.rand(3, generator=g)  # [0.8823, 0.9150, 0.3829]  ← Back to beginning!
```

---

## Multinomial Sampling - Weighted Random Selection

```python
torch.multinomial(p, num_samples=20, replacement=True, generator=g)
```

**What is multinomial sampling?**
- Samples **indices** based on **weighted probabilities**
- Like a weighted lottery where some outcomes are more likely than others

**Parameters:**
- `p` - Probability distribution (weights for each index)
- `num_samples` - How many samples to draw
- `replacement=True` - Can pick the same index multiple times
- `generator=g` - Use seeded generator for reproducibility

**Concrete example:**

```python
# Probabilities for 3 options:
p = tensor([0.1, 0.6, 0.3])
#           10%  60%  30%
#           ↓    ↓    ↓
#          idx0  idx1 idx2

# Sample 20 times:
samples = torch.multinomial(p, num_samples=20, replacement=True)
# Result: tensor([1, 1, 2, 1, 1, 0, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1])

# Count the results:
# Index 0 appears: 1 time   (5%)  ≈ 10% expected
# Index 1 appears: 13 times (65%) ≈ 60% expected  ← Most frequent!
# Index 2 appears: 6 times  (30%) ≈ 30% expected
```

**Applied to character sampling:**

```python
# Get probabilities for characters after '.'
p = N[0].float()
p = p / p.sum()

# Sample next character indices
ix = torch.multinomial(p, num_samples=10, replacement=True, generator=g)
# Result: tensor([1, 13, 1, 19, 11, 1, 8, 1, 14, 5])
#                 ↓   ↓  ↓   ↓   ↓  ↓  ↓  ↓   ↓  ↓
#                 a   m  a   s   k  a  h  a   n  e

# Convert to characters:
for i in ix:
    print(itos[i.item()], end='')
# Output: "amasakane"
```

**Notice:** 'a' appears 4 times because it has high probability (13.77%) - multinomial sampling **respects the learned frequency patterns**!

---

## Putting It Together - Name Generation

```python
g = torch.Generator().manual_seed(1357902468)

for i in range(20):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```

This loop generates 20 new names using the bigram model:

1. **Initialize** - Start with `ix=0` (the '.' token)
2. **Get probabilities** - Convert current character's row to probability distribution
3. **Sample next** - Use multinomial to pick next character based on learned patterns
4. **Append** - Add sampled character to output
5. **Check end** - If we sample '.' again, the name is complete
6. **Print** - Display the generated name

---

## Uniform Distribution vs. Bigram-Trained Model

There are two ways to generate the next character:

### Option 1: Uniform Distribution (Random Baseline)
```python
p = torch.ones(27) / 27  # Every character has equal probability (1/27 ≈ 3.7%)
```

**Result:** Completely random names with no structure
- Example output: `xqzrt.`, `bfjkl.`, `pqwmn.`
- **No learning** - treats all character transitions as equally likely
- Nonsensical - doesn't respect language patterns

### Option 2: Bigram-Trained Model (Learned Patterns)
```python
p = N[ix].float()
p = p / p.sum()
```

**Result:** Names that follow learned bigram statistics
- Example output: `jani.`, `moriah.`, `deliah.`, `kaleigh.`
- **Trained on data** - respects which character transitions are common
- More realistic - uses actual patterns from 32,033 real names

**Key Insight:** The bigram model is trained (albeit only on immediate character pairs), so it generates names that are **significantly better than random**, though still quite simple. It understands:
- Which letters commonly start names
- Which letters commonly follow others ('q' → 'u', 'a' → 'n')
- Which letters commonly end names

However, this model is still **teeny tiny** in terms of sophistication - it only looks at the previous character, not longer patterns or word structure. More advanced models (coming later) will look at longer contexts and produce even more realistic names!

---

## Efficient Vectorized Normalization with Broadcasting

Instead of normalizing rows one at a time in a loop, we can normalize **all rows at once** using PyTorch's broadcasting:

```python
P = N.float()
P = P / P.sum(dim=1, keepdim=True)
```

This is a **vectorized** operation that converts the entire count matrix to probabilities in one efficient step.

---

### Understanding Broadcasting in PyTorch

**Broadcasting Rules:**

Two tensors are broadcastable if, when iterating over dimension sizes **starting from the trailing (rightmost) dimension**, the dimension sizes must either:
1. **Be equal**, or
2. **One of them is 1**, or
3. **One of them does not exist**

---

### Step-by-Step Breakdown

**Step 1: Compute row sums with keepdim**
```python
P.sum(dim=1, keepdim=True)
# Shape: (27, 1) - one sum per row, dimension preserved
```

**Without `keepdim`:** Shape would be `(27,)` - loses the dimension
**With `keepdim`:** Shape is `(27, 1)` - keeps the dimension structure

**Example values:**
```python
P.sum(dim=1, keepdim=True) = [[32033],   # Row 0 sum
                               [15426],   # Row 1 sum
                               [2104],    # Row 2 sum
                               ...]
# Shape: (27, 1)
```

---

**Step 2: Division with broadcasting**
```python
P = P / P.sum(dim=1, keepdim=True)
#   (27, 27) / (27, 1) → broadcasts and performs element-wise division
```

**Broadcasting compatibility check (right to left):**
- **Trailing dimension:** 27 vs 1 → ✅ One is 1, broadcastable
- **Second dimension:** 27 vs 27 → ✅ Equal, broadcastable

**What happens:** Since the shapes are compatible, PyTorch **stretches** the `(27, 1)` tensor to `(27, 27)` by **replicating** each row sum across all 27 columns:

```python
# Original (27, 1):
[[32033],
 [15426],
 [2104],
 ...]

# Broadcasted to (27, 27):
[[32033, 32033, 32033, ..., 32033],  # First sum copied 27 times
 [15426, 15426, 15426, ..., 15426],  # Second sum copied 27 times
 [2104,  2104,  2104,  ..., 2104],   # Third sum copied 27 times
 ...]
```

Then performs **element-wise division**:
```python
P[i, j] = P[i, j] / row_sum[i]  # For every element in the matrix
```

---

### Why Broadcasting is Powerful

**Without broadcasting (slow, manual loop):**
```python
P = N.float()
for i in range(27):
    row_sum = P[i].sum()
    for j in range(27):
        P[i, j] = P[i, j] / row_sum  # Element-by-element division
```

**With broadcasting (fast, vectorized):**
```python
P = N.float()
P = P / P.sum(dim=1, keepdim=True)  # Single operation!
```

**Advantages:**
- ✅ **Concise** - One line instead of nested loops
- ✅ **Fast** - Uses optimized C/CUDA kernels
- ✅ **GPU-friendly** - All operations happen in parallel
- ✅ **Memory efficient** - No actual copying in memory, just clever indexing

**Result:** Every row in `P` now sums to exactly 1.0, making each row a valid probability distribution for sampling the next character!

---

### ⚠️ Critical: Why `keepdim=True` Is Essential

**What happens WITHOUT `keepdim=True`:**

```python
P = N.float()
P = P / P.sum(dim=1)  # ❌ WRONG! Normalizes columns instead of rows
```

**The Problem:**

Without `keepdim=True`, the sum has shape `(27,)` instead of `(27, 1)`:

```python
P.sum(dim=1).shape          # torch.Size([27])
P.sum(dim=1, keepdim=True).shape  # torch.Size([27, 1])
```

**Broadcasting alignment happens from RIGHT to LEFT:**

```python
# WITHOUT keepdim:
P shape:   (27, 27)
sum shape: (27,)

# Align from the right:
# 27, 27
#     27  ← Aligns with the last dimension

# PyTorch implicitly treats (27,) as (1, 27) for broadcasting:
# 27, 27
#  1, 27  ← A dimension of 1 gets added on the LEFT
```

**Result:** Each **column** gets divided by the corresponding sum element, not each **row**!

```python
# What actually happens (WRONG):
P[:, j] = P[:, j] / sum[j]  # Divides column j by sum[j]

# What we wanted (CORRECT):
P[i, :] = P[i, :] / sum[i]  # Divide row i by sum[i]
```

---

**WITH `keepdim=True` (CORRECT):**

```python
# WITH keepdim:
P shape:    (27, 27)
sum shape:  (27, 1)

# Align from the right:
# 27, 27
# 27,  1  ← The 1 broadcasts across the columns
```

**Result:** Each **row** gets divided by its own row sum - exactly what we want!

```python
# What happens (CORRECT):
P[i, :] = P[i, :] / sum[i, 0]  # Divide entire row i by its sum
```

---

### Key Takeaway: Treat Broadcasting with Respect

Broadcasting is powerful but **alignment matters**:

- Broadcasting **always starts from the trailing (rightmost) dimension**
- Without `keepdim`, `(27,)` aligns as `(1, 27)` → normalizes **columns** ❌
- With `keepdim`, `(27, 1)` aligns correctly → normalizes **rows** ✅

**Always verify your broadcasting behavior** by checking shapes:

```python
P.shape           # (27, 27)
P.sum(dim=1, keepdim=True).shape  # (27, 1) ✓ Correct for row normalization
P.sum(dim=0, keepdim=True).shape  # (1, 27) ✓ Would normalize columns instead
```

This subtle difference completely changes what gets normalized!
- ✅ **GPU-friendly** - All operations happen in parallel
- ✅ **Memory efficient** - No actual copying in memory, just clever indexing

**Result:** Every row in `P` now sums to exactly 1.0, making each row a valid probability distribution for sampling the next character!
