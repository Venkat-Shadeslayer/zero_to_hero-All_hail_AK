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

---

## Evaluating Model Quality: Assigned Probabilities

Now that we have a trained bigram model with probabilities, let's examine what probabilities it assigns to actual bigrams from our training data:

```python
for w in words[:3]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        print(f"{ch1},{ch2} -> {prob:.4f}")
```

**Example output for "emma", "olivia", "ava":**
```
.,e -> 0.0478    (4.78% chance '.' is followed by 'e')
e,m -> 0.0377    (3.77% chance 'e' is followed by 'm')
m,m -> 0.0253    (2.53% chance 'm' is followed by 'm')
m,a -> 0.3899    (38.99% chance 'm' is followed by 'a')  ← High!
a,. -> 0.1960    (19.60% chance 'a' is followed by '.')  ← High!
```

---

### What Do These Probabilities Tell Us?

**Baseline (uniform distribution):**
If every bigram was equally likely, we'd expect:
```
P(any bigram) = 1/27 ≈ 0.037 = 3.7%
```

**Our model learns useful patterns:**
- Some transitions are **much more likely**: `m → a` (38.99%) is 10× more likely than uniform
- Some transitions are **moderately likely**: `a → .` (19.60%) is 5× more likely
- Some are **less likely**: `m → m` (2.53%) is slightly below uniform

**This shows the model is learning something useful!** It captures real patterns:
- `'m'` is commonly followed by `'a'` (maria, emma, amanda...)
- Names often end in `'a'` (high probability of `a → .`)
- Double letters like `'mm'` are relatively rare

---

### The Gold Standard: Perfect Model

**What would a perfect model predict?**

A **perfect model** would assign:
- **Probability = 1.0 (100%)** to the actual next character that appears
- **Probability = 0.0 (0%)** to all other characters

**Example:** If the training data has "emma", a perfect model would predict:
```
P(. → e) = 1.0    (absolutely certain 'e' comes next)
P(e → m) = 1.0    (absolutely certain 'm' comes next)
P(m → m) = 1.0    (absolutely certain 'm' comes next)
P(m → a) = 1.0    (absolutely certain 'a' comes next)
P(a → .) = 1.0    (absolutely certain '.' comes next)
```

**On the training set**, a perfect model would "memorize" every transition and predict with 100% confidence.

**Why we don't have this:**
- Our simple bigram model doesn't memorize individual names
- It averages over all training examples
- After seeing "emma" and "ema" and "amanda", when it sees `m`, it splits probability between `m`, `a`, `e`, etc.

---

### Summarizing Model Quality: A Single Number

**The Question:** We have many probabilities for different bigrams. How do we summarize the model's overall quality into **a single understandable number**?

**The Answer:** We'll use the **likelihood** (or more commonly, the **log-likelihood** or **loss**).

---

## Likelihood: Product of Probabilities

The **likelihood** of the data given the model is the **product** of all the probabilities the model assigns to the actual observed transitions:

```
Likelihood = P(bigram_1) × P(bigram_2) × P(bigram_3) × ... × P(bigram_n)
```

**Intuition:** 
- If the model is good, each probability will be **high** (close to 1.0)
- The product of many high probabilities is still reasonably high
- If the model is bad, probabilities will be **low** (close to 0)
- The product becomes **extremely tiny**

**Example for "emma":**
```
Likelihood = P(. → e) × P(e → m) × P(m → m) × P(m → a) × P(a → .)
           = 0.0478 × 0.0377 × 0.0253 × 0.3899 × 0.1960
           = 0.0000034... (very small!)
```

**Problem:** Probabilities are between 0 and 1, so multiplying many of them produces **extremely tiny numbers** that are:
- Hard to interpret
- Numerically unstable (underflow in computers)
- Difficult to optimize

---

## Log Likelihood: Sum of Log Probabilities

Instead of working with products, we take the **logarithm** of the likelihood:

```
Log Likelihood = log(Likelihood)
               = log(P₁ × P₂ × P₃ × ...)
               = log(P₁) + log(P₂) + log(P₃) + ...    ← Product becomes sum!
```

**Key property:** `log(a × b × c) = log(a) + log(b) + log(c)`

```python
log_likelihood = 0.0
n = 0

for w in words[:3]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1},{ch2} -> {prob:.4f} {logprob:.4f}")
```

**Example output:**
```
.,e -> 0.0478 -3.0408
e,m -> 0.0377 -3.2793
m,m -> 0.0253 -3.6772
m,a -> 0.3899 -0.9418   ← High prob → log close to 0
a,. -> 0.1960 -1.6299
```

---

### Understanding Log of Probabilities

**Key facts about logarithms:**
- `log(1.0) = 0`    ← Perfect prediction
- `log(0.5) ≈ -0.69`  ← Moderate prediction
- `log(0.1) ≈ -2.30`  ← Poor prediction
- `log(0.01) ≈ -4.61` ← Very poor prediction

**Since probabilities are between 0 and 1:**
- `log(probability)` is **always negative** (or zero if prob = 1)
- Higher probability → log value closer to 0 (less negative)
- Lower probability → log value more negative

**In our example:**
- `m → a` has prob 0.3899 → log = -0.9418 (close to 0, good!)
- `o → .` has prob 0.0123 → log = -4.3982 (very negative, poor!)

---

## Negative Log Likelihood (NLL): The Loss Function

**Problem:** Higher log likelihood is better, but in machine learning we prefer **loss functions** where **lower is better**.

**Solution:** Take the **negative** of the log likelihood:

```python
print(f"Log-likelihood: {log_likelihood:.4f}")
nll = -log_likelihood
print(f"Negative Log-likelihood: {nll:.4f}")
```

**Output:**
```
Log-likelihood: -38.7858
Negative Log-likelihood: 38.7858
```

**Now:**
- **Lower NLL = Better model** (less loss)
- **Higher NLL = Worse model** (more loss)

This aligns with standard optimization: **minimize the loss**.

---

## Per-Character Negative Log Likelihood (Normalized Loss)

For fair comparison across sequences of different lengths, we **average** the NLL by dividing by the number of characters:

```python
print(f"Per-character Negative Log-likelihood: {nll/n:.4f}")
```

**Output:**
```
Per-character Negative Log-likelihood: 2.4241
```

**Why normalize?**
- A longer name naturally has higher total NLL (more bigrams to predict)
- Averaging gives us a **per-character** measure
- Now we can compare models fairly regardless of sequence length

**Interpretation:**
- **2.4241** is our model's average loss per character
- Perfect model would have NLL = 0 (always predicts correctly with prob = 1.0)
- Random guessing (uniform) would have NLL ≈ 3.30 (since log(1/27) ≈ -3.30)
- Our model (2.4241) is better than random but far from perfect

---

## Testing on Individual Names

Let's test the model on specific names to understand what the NLL metric means:

**Example 1: "andrek"**
```
NLL ≈ 3.03 per character
```

This is **slightly worse** than the average (2.4241), meaning:
- The bigrams in "andrek" are somewhat less common in the training data
- The model is less confident about these transitions
- Still reasonable - the model has seen these patterns before

**Example 2: "andrejq"**
```
NLL = ∞ (infinity)
```

This is **catastrophically bad** because:
- The bigram `'j' → 'q'` has **zero probability** (never seen in training)
- `P[ix_j, ix_q] = 0.0`
- `log(0) = -∞`
- Negative log likelihood = `-(-∞) = ∞`

---

### The Zero Probability Problem

**Critical issue:** If our model assigns **probability 0** to any bigram that actually occurs, the loss becomes **infinite**.

**Why this happens:**
```python
# If 'jq' never appears in training data:
N[ix_j, ix_q] = 0

# After normalization:
P[ix_j, ix_q] = 0.0 / sum = 0.0

# Taking log:
log(0.0) = -∞

# Negative log likelihood:
NLL = -(-∞) = ∞
```

**Consequences:**
- Model completely fails on unseen bigrams
- Cannot evaluate on names with rare character combinations
- Numerically unstable (infinity breaks optimization)
- Unrealistic - even unlikely bigrams should have *some* tiny probability

**Real-world example:**
- Training data might not contain "queen" or "quiz"
- But `'q' → 'u'` is a valid English pattern
- Model assigns 0% probability → infinite loss
- Model claims these names are "impossible"

---

### Why This Motivates Smoothing

This zero-probability problem highlights a major limitation:
- **Raw frequency counts** lead to overconfident predictions
- Unseen bigrams get probability 0 (too harsh!)
- Model cannot generalize to slightly unusual patterns

**Solution (coming next):** Add **smoothing** by adding a small constant to all counts before normalization. This ensures:
- Every bigram has at least a tiny non-zero probability
- Model is less overconfident
- Can handle unseen combinations gracefully
- Loss remains finite even for rare patterns

**For now, key takeaway:**
- Average NLL = 2.4241 is our baseline
- Names with common patterns have NLL < 2.4
- Names with rare patterns have NLL > 2.4
- Names with impossible patterns have NLL = ∞ (model breaks!)

---

## Model Smoothing: Adding Fake Counts

To fix the zero-probability problem, we use a simple but powerful technique: **add a small constant to all counts before normalization**.

```python
P = (N + 1).float()  # Add 1 to every count
P /= P.sum(dim=1, keepdim=True)
```

This is called **Laplace smoothing** or **additive smoothing** (also known as "add-one smoothing" when adding 1).

---

### How It Works

**Before smoothing:**
```python
N[ix_j, ix_q] = 0  # Never seen in training
P[ix_j, ix_q] = 0 / row_sum = 0.0  # Zero probability
log(0.0) = -∞  # Infinite loss!
```

**After smoothing:**
```python
(N + 1)[ix_j, ix_q] = 0 + 1 = 1  # Now has a fake count
P[ix_j, ix_q] = 1 / row_sum ≈ 0.001  # Tiny but non-zero probability
log(0.001) ≈ -6.9  # Finite loss!
```

**What happens:**
- **Every bigram** now has at least count = 1 (the fake count)
- **Previously unseen bigrams** go from 0 → 1
- **Previously seen bigrams** increase slightly (e.g., 100 → 101)
- After normalization, all probabilities are **non-zero**

---

### The Smoothing Trade-off

**Amount of smoothing controls model behavior:**

**Add more (e.g., +10, +100):**
- Model becomes **smoother** (more uniform)
- All bigrams get more equal probabilities
- Less confident predictions
- Better generalization to unseen patterns
- But may ignore training data patterns

**Add less (e.g., +0.1, +0.01):**
- Model stays **peakier** (closer to original)
- Strong distinctions between common and rare bigrams
- More confident predictions
- Stays closer to training data
- Better fit to training data

**Add exactly 1:**
- Balanced middle ground
- Standard Laplace smoothing
- Prevents zeros while preserving most patterns

---

### Effect on Model Quality

**Without smoothing:** `P = N.float()`
- NLL = 2.4241 on training data
- NLL = ∞ on "andrejq" (contains unseen 'jq')

**With smoothing:** `P = (N + 1).float()`
- NLL ≈ 2.45 on training data (slightly higher)
- NLL ≈ 8-10 on "andrejq" (high but finite!)

**Trade-off:**
- Training data fit gets **slightly worse** (2.42 → 2.45)
- Unseen patterns become **possible** (∞ → finite)
- Model is more **robust** and **generalizable**

---

### Visualizing the Effect

**Example row (character 'j'):**

| Next char | Count | Prob (no smooth) | Prob (smooth +1) |
|-----------|-------|------------------|------------------|
| 'a' | 1084 | 0.741 | 0.724 |
| 'e' | 173 | 0.118 | 0.116 |
| '.' | 82 | 0.056 | 0.055 |
| 'q' | **0** | **0.000** | **0.0007** |
| ... | ... | ... | ... |

**Notice:**
- Common transitions ('a', 'e') barely change
- Impossible transition ('q') now has tiny probability
- Model is **less extreme** but still respects patterns

---

### Why This Is Important

**In practice:**
- Real-world test data **always** contains patterns not in training
- Zero probabilities → model crashes
- Smoothing → model degrades gracefully
- Small amount of smoothing rarely hurts, often helps

**The smoothing parameter** (how much to add) is a **hyperparameter** you can tune:
- Too much → model ignores training data
- Too little → model overfits training data
- Typical values: 0.01 to 10

**For our bigram model:**
- Adding 1 is reasonable (Laplace smoothing)
- Trades tiny loss increase (2.42 → 2.45) for robustness
- Can now handle any name, even with rare bigrams!

---

# Neural Network Approach: Learning the Same Thing

Now we'll take a **neural network approach** to arrive at the same bigram model, but through gradient-based optimization instead of counting.

---

## Creating the Training Dataset

We need to construct a dataset of **input-output pairs** from our bigrams:

```python
# Create the training set of all bigrams (x, y)
xs, ys = [], []

for w in words[:1]:  # Start with one name to understand
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

**Example with the first name "emma":**

| Bigram | Input (x) | Target (y) |
|--------|-----------|------------|
| . → e  | 0         | 5          |
| e → m  | 5         | 13         |
| m → m  | 13        | 13         |
| m → a  | 13        | 1          |
| a → .  | 1         | 0          |

**Result:**
```python
xs = tensor([0, 5, 13, 13, 1])  # Input characters
ys = tensor([5, 13, 13, 1, 0])  # Target characters (what should come next)
```

---

## The Neural Network Goal

**Objective:** Train a neural network that, when given the input characters `xs`, predicts high probabilities for the corresponding target characters `ys`.

**Input-Output Mapping:**
- Given `x = 0` (character '.'), predict high probability for `y = 5` (character 'e')
- Given `x = 5` (character 'e'), predict high probability for `y = 13` (character 'm')
- Given `x = 13` (character 'm'), predict high probability for `y = 13` (character 'm')
- And so on...

**Key Insight:** We're treating letters as **integers** (indices), which will later become **embeddings** in the neural network. Each character is represented by a number (0-26), and the network will learn patterns from these numerical representations.

---

## What We'll Learn

**Instead of explicitly counting and normalizing:**
1. We'll initialize random "weights" (parameters)
2. Forward pass: compute predictions for each input
3. Compute loss: how far are predictions from targets?
4. Backward pass: compute gradients
5. Update weights to reduce loss
6. Repeat until converged

**The beauty:** The neural network should learn the same probability distribution that we derived from counting, but through gradient descent!

---

## ⚠️ Important Note: torch.tensor vs torch.Tensor

There are **two ways** to construct a tensor in PyTorch:

### `torch.tensor()` (lowercase 't')
```python
xs = torch.tensor([0, 5, 13])
```
- **Infers dtype** from the data automatically
- `[0, 5, 13]` → `dtype=torch.int64` (long)
- `[0.5, 1.2]` → `dtype=torch.float32` (float)
- **Preferred** for creating tensors from Python lists/arrays

### `torch.Tensor()` (uppercase 'T')
```python
xs = torch.Tensor([0, 5, 13])
```
- **Always creates float32** regardless of input
- `[0, 5, 13]` → `dtype=torch.float32`
- Legacy constructor, aliased to `torch.FloatTensor`
- Can cause unexpected behavior with integer data

**Example difference:**
```python
a = torch.tensor([1, 2, 3])     # dtype: torch.int64
b = torch.Tensor([1, 2, 3])     # dtype: torch.float32

print(a.dtype)  # torch.int64 ✓
print(b.dtype)  # torch.float32 (unexpected for integers!)
```

**Best practice:** Use `torch.tensor()` (lowercase) for explicit control and predictable behavior. Always check the PyTorch docs when in doubt!

---

**Coming up next:** We'll build a simple neural network to learn the bigram probabilities from scratch, using only the training pairs (xs, ys) and gradient descent!

---

## Input Encoding: One-Hot Representation

**Problem:** We have integer indices like `xs = [0, 5, 13, 13, 1]`, but we can't just plug raw integers directly into a neural network. We need to convert them to a format the network can understand.

**Solution:** Use **one-hot encoding** to represent each character as a vector.

---

### What is One-Hot Encoding?

**One-hot encoding** converts an integer index into a **binary vector**:
- Vector has length equal to the number of possible values (27 in our case)
- All elements are 0 except for one position
- The position corresponding to the index is set to 1

**Example:**

For the character 'e' (index 5):
```
Index: 5
One-hot: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          ↑  ↑  ↑  ↑  ↑  ↑  position 5 is 1, rest are 0
          0  1  2  3  4  5
```

For the character 'm' (index 13):
```
Index: 13
One-hot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                   ↑  position 13 is 1, rest are 0
```

---

### Converting Our Training Data

**Original data (indices):**
```python
xs = tensor([0, 5, 13, 13, 1])  # 5 examples
```

**After one-hot encoding:**
Each integer becomes a 27-dimensional vector, so we get a **5 × 27 matrix**:

```
Example 0 (char '.'): [1, 0, 0, 0, 0, 0, ..., 0]  ← position 0 is 1
Example 1 (char 'e'): [0, 0, 0, 0, 0, 1, ..., 0]  ← position 5 is 1
Example 2 (char 'm'): [0, 0, 0, 0, 0, 0, ..., 0]  ← position 13 is 1
Example 3 (char 'm'): [0, 0, 0, 0, 0, 0, ..., 0]  ← position 13 is 1
Example 4 (char 'a'): [0, 1, 0, 0, 0, 0, ..., 0]  ← position 1 is 1
```

**Shape:** `(5, 27)` - 5 examples, each with 27 features (one per possible character)

---

### Why One-Hot Encoding?

**Advantages:**
1. **No ordinality** - Treats each character as equally different
   - Index 5 (e) is not "5 times more" than index 1 (a)
   - No artificial ordering or magnitude relationships
2. **Direct lookup** - Each character activates exactly one neuron
3. **Easy to interpret** - Clear which character is which
4. **Standard input format** - Neural networks expect real-valued vectors

**Disadvantage:**
- **High dimensionality** - 27 dimensions for 27 characters (sparse, mostly zeros)
- **Memory inefficient** - Most values are 0

**Note:** Later, we'll use **embeddings** which are a more efficient learned representation, but one-hot encoding is a good starting point!

---

### PyTorch One-Hot Encoding

```python
import torch.nn.functional as F

# Convert integer indices to one-hot vectors
xenc = F.one_hot(xs, num_classes=27).float()  # Convert to float!
```

**Example:**
```python
xs = tensor([0, 5, 13, 13, 1])

xenc = F.one_hot(xs, num_classes=27).float()
# Shape: (5, 27)
# xenc[0] = [1., 0., 0., ..., 0.]  # One-hot for index 0
# xenc[1] = [0., 0., 0., 0., 0., 1., 0., ...]  # One-hot for index 5
# xenc[2] = [0., 0., 0., 0., 0., 0., ..., 0., 1., ...]  # One-hot for index 13
```

---

### Important: Data Type Conversion

**By default, `F.one_hot()` returns integers:**
```python
xenc = F.one_hot(xs, num_classes=27)
print(xenc.dtype)  # torch.int64 ❌ Not suitable for neural networks!
```

**Neural networks require floating point:**
```python
xenc = F.one_hot(xs, num_classes=27).float()
print(xenc.dtype)  # torch.float32 ✓ Ready for neural networks!
```

**Why floating point is required:**
1. **Neural network weights** are float32/float64
2. **Matrix multiplication** between input and weights requires matching types
3. **Gradient computation** works with floating point values
4. **Backpropagation** requires continuous, differentiable operations
5. **Optimization algorithms** (SGD, Adam, etc.) update weights as floats

**What happens if you forget `.float()`:**
```python
# ❌ Type mismatch error:
weights = torch.randn(27, 27)  # float32
xenc = F.one_hot(xs, num_classes=27)  # int64
result = xenc @ weights  # ERROR: int64 @ float32 not supported!

# ✓ Correct:
xenc = F.one_hot(xs, num_classes=27).float()  # float32
result = xenc @ weights  # Works! float32 @ float32
```

**Best practice:** Always convert one-hot encoded tensors to float before feeding into neural networks.

**Result:** `xenc` is now ready to be fed into a neural network as input!

---

## Building the Neural Network Forward Pass

Now we'll create a simple neural network to predict the next character. We'll use a single-layer network (just matrix multiplication).

### Step 1: Initialize Random Weights

```python
W = torch.randn((27, 27))
```

This creates a **27 × 27 weight matrix** with random values. Each element is sampled from a standard normal distribution (mean=0, std=1).

**Why this shape?**
- Input: 27-dimensional one-hot vector (one per character)
- Output: 27 values (one score per possible next character)
- Weight matrix connects all inputs to all outputs

---

### Step 2: Forward Pass - Matrix Multiplication

```python
xenc @ W  # Matrix multiplication
```

**Shape analysis:**
```
xenc:  (5, 27)   # 5 examples, each with 27 features
W:     (27, 27)  # Weights connecting 27 inputs to 27 outputs
Result: (5, 27)  # 5 examples, each with 27 output scores
```

**For our "emma" example with 5 bigrams:**
- Row 0: Scores for what follows '.'
- Row 1: Scores for what follows 'e'
- Row 2: Scores for what follows 'm' (first occurrence)
- Row 3: Scores for what follows 'm' (second occurrence)
- Row 4: Scores for what follows 'a'

**These raw outputs are called "logits"** - they're not probabilities yet!

---

### Step 3: From Logits to Probabilities

**Problem:** Neural network outputs (logits) are **raw real numbers**:
- Can be negative or positive
- Don't sum to 1
- Not valid probabilities

**Solution:** Apply the same transformation we used for count matrix:

```python
logits = xenc @ W              # Raw scores from neural network
counts = logits.exp()          # Exponentiate to make positive
probs = counts / counts.sum(1, keepdim=True)  # Normalize rows to sum to 1
```

**Step-by-step transformation:**

**Before (logits - can be anything):**
```
logits[0] = [-2.3, 0.5, 1.8, -0.9, ...]  # Raw scores, can be negative
```

**After exp() (counts - all positive):**
```
counts[0] = [0.10, 1.65, 6.05, 0.41, ...]  # All positive, like counts!
```

**After normalization (probabilities - sum to 1):**
```
probs[0] = [0.01, 0.19, 0.71, 0.05, ...]  # Sum = 1.0 ✓
```

---

### Why Exponentiation?

**The `exp()` function serves multiple purposes:**

1. **Makes everything positive** - probabilities must be non-negative
   ```
   exp(-5) ≈ 0.007  (small but positive)
   exp(0) = 1.0
   exp(5) ≈ 148.4   (large and positive)
   ```

2. **Preserves relative ordering** - larger logits → larger probabilities
   ```
   If logit_a > logit_b, then exp(logit_a) > exp(logit_b)
   ```

3. **Smooth and differentiable** - critical for gradient-based learning

4. **Amplifies differences** - exponentiation emphasizes stronger predictions
   ```
   logits:  [1.0, 2.0, 3.0]
   exp:     [2.7, 7.4, 20.1]  ← Differences magnified
   ```

---

### Connecting to the Count-Based Approach

**Count-based model (what we did before):**
```python
N = count_bigrams()          # Count occurrences
P = N.float()                # Convert to float
P /= P.sum(1, keepdim=True)  # Normalize rows
```

**Neural network model (what we're doing now):**
```python
logits = xenc @ W            # Compute scores
counts = logits.exp()        # Make positive (like raw counts)
probs = counts / counts.sum(1, keepdim=True)  # Normalize rows
```

**Key insight:** The neural network is trying to **learn** the same probability distribution that we derived from counting! The weight matrix `W` will be optimized to produce probabilities similar to our count-based model.

---

### Recap for "emma" (5 bigrams)

We have **5 input-output training examples** from "emma":

| Example | Input char | Target char | One-hot input | Network output |
|---------|-----------|-------------|---------------|----------------|
| 0 | '.' (0) | 'e' (5) | xenc[0] | probs[0] (27 values) |
| 1 | 'e' (5) | 'm' (13) | xenc[1] | probs[1] (27 values) |
| 2 | 'm' (13) | 'm' (13) | xenc[2] | probs[2] (27 values) |
| 3 | 'm' (13) | 'a' (1) | xenc[3] | probs[3] (27 values) |
| 4 | 'a' (1) | '.' (0) | xenc[4] | probs[4] (27 values) |

**Goal:** Train the network so that `probs[i, ys[i]]` is high for all examples.
- Example 0: We want `probs[0, 5]` to be high (predict 'e' after '.')
- Example 1: We want `probs[1, 13]` to be high (predict 'm' after 'e')
- And so on...

**Shape of outputs:**
```
probs.shape = (5, 27)  # 5 examples × 27 probability distributions
```

Each row is a probability distribution over the 27 possible next characters!

## The Big Picture: Maximum Likelihood Training

**GOAL:** Maximize likelihood of the data with respect to model parameters (statistical modeling)

**Equivalences:**
1. **Maximize likelihood** 
   ↓ (take log, which is monotonic)
2. **Maximize log likelihood**
   ↓ (multiply by -1)
3. **Minimize negative log likelihood**
   ↓ (divide by n)
4. **Minimize average negative log likelihood**

All four objectives are **equivalent** for optimization!

**Key property:** `log(a × b × c) = log(a) + log(b) + log(c)`
- Products (likelihood) become sums (log likelihood)
- Sums are easier to work with computationally
- Gradients are nicer for optimization

**In practice:** We always minimize the (average) negative log likelihood, also called the **cross-entropy loss** in classification tasks.

**Our result:** Per-character NLL = 2.4241
- This is our baseline bigram model performance
- Lower values = better model
- We'll use this metric to evaluate improvements!

**Coming next:** We'll learn to compute a single number that tells us:
- **Higher is better** (likelihood) or **lower is better** (loss)
- How well the model predicts the training data overall
- Whether our model is improving as we make changes

This single metric will be crucial for training neural network models later!

---

## Extracting Specific Bigram Probabilities from the Neural Network

Now let's examine the **exact probabilities** that our neural network assigns to the bigrams in the first training example, "emma":

```python
probs[0,5], probs[1,13], probs[2,13], probs[3,1], probs[4,0]
```

**What do these indices represent?**

Each index pair `[row, column]` corresponds to a specific bigram transition:

| Index | Bigram | Meaning | Probability |
|-------|--------|---------|-------------|
| `probs[0, 5]` | `.` → `e` | Start token to 'e' | `probs[0, 5]` |
| `probs[1, 13]` | `e` → `m` | 'e' followed by 'm' | `probs[1, 13]` |
| `probs[2, 13]` | `m` → `m` | 'm' followed by 'm' | `probs[2, 13]` |
| `probs[3, 1]` | `m` → `a` | 'm' followed by 'a' | `probs[3, 1]` |
| `probs[4, 0]` | `a` → `.` | 'a' to end token | `probs[4, 0]` |

**Together, these five probabilities represent:**
- How likely our neural network thinks each transition in "emma" is
- The network's "confidence" in predicting each next character
- The values we'll use to compute the loss for this example

**Key insight:** 
- **Row index** = current character (input to network)
- **Column index** = next character (what we're predicting)
- **Value** = probability assigned by the network to this transition

**Example breakdown for "emma":**
```python
# Input: xs = [0, 5, 13, 13, 1]  →  [., e, m, m, a]
# Label: ys = [5, 13, 13, 1, 0]  →  [e, m, m, a, .]

# For bigram 0: (. → e)
probs[0, 5]   # Row 0 = '.', Column 5 = 'e'

# For bigram 1: (e → m)
probs[1, 13]  # Row 5 = 'e', Column 13 = 'm'
# Note: We use row 1 because this is the 2nd example in our batch

# For bigram 2: (m → m)
probs[2, 13]  # Row 13 = 'm', Column 13 = 'm'

# For bigram 3: (m → a)
probs[3, 1]   # Row 13 = 'm', Column 1 = 'a'

# For bigram 4: (a → .)
probs[4, 0]   # Row 1 = 'a', Column 0 = '.'
```

These specific probabilities are exactly what we need to:
1. **Compute the loss** for this training example
2. **Calculate gradients** to improve the model
3. **Evaluate** how well the network learned this name

The **higher** these probabilities, the **better** our model is at predicting "emma"!

---

## Scaling Up: Training on the Full Dataset

Now that we understand the mechanics, let's train our neural network on **all 32,033 names** instead of just the first one!

### Cell 1: Creating the Complete Training Dataset

```python
# create the dataset
xs, ys = [], []
for w in words:  # ← Note: Using ALL words now, not just words[:1]
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

**Output:**
```
number of examples: 228,146
```

**What changed from before:**

| Before (Single Word) | Now (Full Dataset) |
|---------------------|-------------------|
| `for w in words[:1]` | `for w in words` |
| 5 training examples | 228,146 training examples |
| Only "emma" bigrams | All bigrams from 32,033 names |

**Key steps:**

1. **Loop through ALL words** - Process every name in the dataset
2. **Extract bigrams** - For each name, create character pairs with start/end tokens
3. **Convert to indices** - Map characters to integers using `stoi`
4. **Create tensors** - Convert lists to PyTorch tensors for efficient computation
5. **Count examples** - `num = xs.nelement()` gives us 228,146 total bigrams
6. **Initialize weights** - Random 27×27 matrix with `requires_grad=True` for training

**Why this matters:**
- More data = better learning
- The network will see diverse patterns across all names
- Loss will be computed across the entire dataset, not just one word

---

### Cell 2: Gradient Descent Training Loop

```python
# gradient descent
for k in range(100):
    
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())
    
    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()
    
    # update
    W.data += -50 * W.grad
```

**This is a complete training loop!** Let's break down what happens in each of the 100 iterations:

---

### The Three Phases of Each Iteration

#### 1️⃣ **Forward Pass** (Prediction)

```python
xenc = F.one_hot(xs, num_classes=27).float()
```
- Convert input indices to one-hot vectors
- Shape: (228146, 27) - one row per training example

```python
logits = xenc @ W
```
- Multiply one-hot encoding by weight matrix
- Each row gets the weights corresponding to its input character
- Shape: (228146, 27) - predictions for all examples

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
```
- Convert logits to counts (exponentiate)
- Normalize to probabilities (each row sums to 1.0)
- Shape: (228146, 27) - probability distribution for each example

```python
loss = -probs[torch.arange(num), ys].log().mean()
```
- Extract the probability assigned to the **correct** character for each example
- Take negative log (higher probability → lower loss)
- Average across all 228,146 examples
- **Single number** that tells us how well the model is doing

---

#### 2️⃣ **Backward Pass** (Compute Gradients)

```python
W.grad = None  # Clear previous gradients
loss.backward()  # Compute ∂loss/∂W for all weights
```

- PyTorch automatically computes how each weight affects the loss
- `W.grad` now contains the gradient (direction of steepest increase)
- Gradients tell us: "Change this weight by X to reduce the loss"

---

#### 3️⃣ **Update** (Improve Weights)

```python
W.data += -50 * W.grad
```

- **Learning rate = 50** - How big of a step to take
- **Negative gradient** - Move in the direction that **reduces** loss
- Update formula: `new_weight = old_weight - learning_rate × gradient`

**Why negative?**
- Gradient points toward **increasing** loss (uphill)
- We want to **decrease** loss (downhill)
- So we move in the **opposite direction** (multiply by -1)

---

### What Loss Should We Achieve?

**The big question:** What's our target loss?

**Answer:** Approximately **2.45** 🎯

**Why?**

Remember earlier when we calculated the bigram model loss using the count matrix `N` with smoothing?

```python
P = (N+1).float()
P /= P.sum(dim=1, keepdim=True)
# Computed loss across all bigrams: ~2.45
```

That bigram model with add-1 smoothing achieved a loss of **~2.45**.

**Our neural network is learning to replicate that same model!**

- Same architecture: bigram (current char → next char)
- Same data: all 228,146 character transitions
- Same objective: minimize negative log likelihood

**Expected outcome after 100 iterations:**
```
Iteration 1:   loss ≈ 3.77  (random initialization)
Iteration 10:  loss ≈ 2.80  (getting better...)
Iteration 50:  loss ≈ 2.50  (close!)
Iteration 100: loss ≈ 2.45  (converged!)
```

**Wink wink wowww!** 🎉 The neural network, through gradient descent, discovers the same solution that we computed analytically with the count matrix!

---

### Key Insights

**1. Neural Network = Differentiable Count Matrix**
- The bigram model counts transitions and normalizes
- The neural network learns weights that encode the same information
- Gradient descent finds these weights automatically

**2. Why the losses match:**
- Both models see the exact same training data
- Both use the same objective (negative log likelihood)
- Both have the same capacity (27×27 parameters)
- The neural network converges to the optimal solution

**3. The power of gradient descent:**
- We didn't hand-code the solution (count and normalize)
- We just defined a loss function and let optimization find the weights
- This scales to much more complex models where analytical solutions don't exist!

**4. Learning rate matters:**
- Too small (e.g., 0.1): slow convergence, needs many iterations
- Just right (e.g., 50): converges quickly to optimal solution
- Too large (e.g., 1000): might overshoot and diverge

---

### Important Note: Smoothing vs. Regularization

**Both approaches have an equivalent concept for controlling probability distribution smoothness!**

#### Count-Based Approach: Smoothing

```python
P = (N + α).float()  # Add smoothing constant α
P /= P.sum(dim=1, keepdim=True)
```

**Effect of smoothing constant:**
- `α = 1` (add-1 smoothing): Prevents zero probabilities, slight smoothing
- `α = 10`: More smoothing, probabilities become more uniform
- `α = 1000` (large number): **Overwhelms actual counts** → Nearly uniform distribution

**Why?** When α is huge:
```python
P[i, j] = (count[i,j] + 1000) / sum_of_row
# If counts are small (e.g., 0-100), adding 1000 dominates
# All entries ≈ 1000, so probabilities ≈ 1/27 (uniform)
```

---

#### Gradient-Based Approach: Initialization (Regularization)

```python
W = torch.zeros((27, 27), requires_grad=True)  # All zeros
```

**What happens with zero initialization?**

1. **Logits become all zeros:**
   ```python
   logits = xenc @ W  # Matrix of zeros
   # Result: all logits = 0
   ```

2. **Counts become all ones:**
   ```python
   counts = logits.exp()  # exp(0) = 1
   # Result: all counts = 1
   ```

3. **Probabilities become uniform:**
   ```python
   probs = counts / counts.sum(1, keepdim=True)
   # Each row: [1, 1, 1, ..., 1] / 27 = [1/27, 1/27, ..., 1/27]
   # Result: uniform distribution! 🎯
   ```

**This is equivalent to very large smoothing (α → ∞) in the count-based approach!**

---

#### The Parallel Concept: Regularization

| Count-Based | Gradient-Based |
|-------------|----------------|
| Smoothing constant (α) | Weight initialization / regularization |
| Large α → uniform probs | W = zeros → uniform probs |
| Controls how much to trust data | Controls starting point of optimization |
| Prevents overfitting to rare bigrams | Prevents overfitting during training |

**Key insight:** 
- **Small/random W** → Model starts with weak beliefs, learns from data
- **W = 0** → Model starts with uniform beliefs (maximum uncertainty)
- **Large smoothing α** → Forces model toward uniform (ignores rare events)

Both are forms of **regularization** - techniques to prevent the model from overfitting to the training data!

**In practice:**
- Random initialization (what we used) is standard for neural networks
- Starting from W=0 would give uniform predictions initially, then gradually learn
- Both converge to similar solutions if the model has enough capacity
- Regularization techniques (like adding penalties to large weights) achieve similar smoothing effects during training

This connection between classical statistical techniques (smoothing) and modern deep learning (initialization/regularization) shows how neural networks are learning the same fundamental principles, just through a different mathematical framework! 🔄

---

### L2 Regularization: Explicit Penalty on Weight Magnitude

Beyond initialization, we can **explicitly** regularize during training by adding a penalty term to the loss:

```python
# Compute mean squared weight
regularization_loss = (W**2).mean()

# Modified loss with L2 regularization
loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
       ├─────────────────────────────────────┘       └──────┬──────────┘
       Data fitting term (NLL)                       Regularization term
```

**The 0.01 is the regularization coefficient (λ)** - it controls the strength of the penalty.

---

#### Two Competing Objectives

The optimization now balances **two goals simultaneously**:

1. **Fit the data** (minimize negative log likelihood)
   - Make predictions accurate
   - Assign high probability to actual next characters
   - Lower is better

2. **Keep weights small** (minimize weight magnitude)
   - Drive W toward zero
   - Prefer simple/smooth solutions
   - Lower is better

**The tradeoff:**

```python
Total Loss = Data Loss + λ × Regularization Loss
           = NLL      + 0.01 × (W²).mean()
```

**What happens during optimization:**
- Gradient descent tries to minimize **both** terms
- If W ≠ 0, we feel a loss! The penalty increases
- Larger weights → larger penalty (quadratic: W²)
- The model must balance: "Is this weight worth the penalty?"

---

#### Effect of Regularization Coefficient (λ)

| λ value | Effect | Behavior |
|---------|--------|----------|
| **λ = 0** | No regularization | Fit data perfectly, may overfit |
| **λ = 0.01** | Mild regularization | Slight smoothing, better generalization |
| **λ = 0.1** | Moderate regularization | More smoothing, simpler model |
| **λ = 10** | Strong regularization | Heavy smoothing, nearly uniform |

**Example with λ = 0.01:**

```python
# Some bigram has strong evidence: count = 100
# Without regularization: W[i,j] might be large (e.g., 5.0)
# Loss: only data term matters

# With λ = 0.01:
# Data term says: "Make W[i,j] large to fit this bigram!"
# Regularization term says: "Keep W[i,j] small to reduce penalty!"
# Final W[i,j]: compromise value (e.g., 4.2)
```

---

#### Mathematical Gradient Decomposition

When we compute gradients with regularization:

```python
loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
loss.backward()
```

The gradient of W has two components:

```
∂loss/∂W = ∂(data_loss)/∂W + ∂(regularization_loss)/∂W
         = gradient_from_data + 0.01 × 2W
         = gradient_from_data + 0.02W
```

**Update step:**
```python
W.data += -50 * W.grad
        = -50 * (gradient_from_data + 0.02W)
        = -50 * gradient_from_data - W
```

**Interpretation:**
- First term: move W to fit the data better
- Second term: **shrink W toward zero** by a small amount
- Every update step pulls W closer to zero while also fitting data!

This is called **weight decay** - weights naturally decay toward zero unless the data provides strong evidence to keep them large.

---

#### Connection to Smoothing (Again!)

**L2 regularization is the neural network equivalent of smoothing!**

| Count-Based Smoothing | Neural Network L2 Regularization |
|----------------------|----------------------------------|
| `P = (N + α) / sum` | `loss = NLL + λ(W²)` |
| Adds constant to all counts | Penalizes large weights |
| Large α → uniform distribution | Large λ → weights near zero → uniform |
| Prevents zero probabilities | Prevents overfitting |
| Static: applied once | Dynamic: continuous pressure during training |

**Both achieve the same goal:** Don't trust rare events too much, smooth the distribution!

---

#### Practical Benefits

**Why add this penalty?**

1. **Prevents overfitting** - Model won't memorize rare bigrams
2. **Better generalization** - Smoother probabilities work better on new names
3. **Numerical stability** - Prevents weights from exploding
4. **Implicit bias** - Favors simpler explanations (Occam's Razor)

**The beauty of regularization:**
```
Without regularization: "Fit the training data perfectly!"
With regularization: "Fit the training data, but keep it simple!"
```

**Result:**
- Training loss might be slightly higher (we're fighting two objectives)
- **Test/validation loss** is often lower (better generalization!)
- Model is more robust to variations in data

This is one of the most important techniques in machine learning - we'll see it everywhere from simple bigram models to GPT-4! 🎯

---

**The beauty:** We've just trained our first neural network! It's simple (single-layer, bigram), but it demonstrates all the key concepts:
- Forward pass (prediction)
- Loss computation (how wrong are we?)
- Backward pass (how to improve?)
- Weight updates (gradient descent)
- **Regularization** (prevent overfitting!)

Next, we'll scale this to deeper networks that can capture longer-range dependencies! 🚀

---

## Why This Makes Perfect Sense (And Why It's Revolutionary!)

### The Results Match - And They Should!

**Of course we got the same loss (~2.45) as the count-based approach!**

Both methods use **exactly the same information:**
- ✅ Look at the **previous character**
- ✅ Predict the **next character**
- ✅ Train on the same 228,146 bigram examples
- ✅ Minimize the same objective (negative log likelihood)

**We didn't add any extra information**, so the neural network converged to the same solution as the analytical approach. It's like arriving at the same destination via two different routes:

| Count-Based Approach | Neural Network Approach |
|---------------------|------------------------|
| Count bigrams → Normalize | Random weights → Gradient descent |
| Direct calculation | Iterative optimization |
| **Loss: ~2.45** | **Loss: ~2.45** |

---

### But Here's Why the Gradient-Based Approach is Revolutionary! 🚀

**The true power isn't in matching the simple bigram model - it's in what comes next!**

The gradient-based approach is **infinitely more flexible**:

#### 🔹 **Same Framework, Unlimited Complexity**

With the count-based method, we're stuck with bigrams. But with gradient descent:

```python
# Simple bigram (what we just did)
logits = xenc @ W  # One layer, previous char → next char

# ⬇️ But we can easily extend to...

# Multi-layer network (look at more context)
hidden = xenc @ W1  # First layer
logits = hidden @ W2  # Second layer

# ⬇️ Or even more layers...

# Deep network (capture complex patterns)
h1 = relu(xenc @ W1)
h2 = relu(h1 @ W2)
h3 = relu(h2 @ W3)
logits = h3 @ W4

# ⬇️ All the way to...

# TRANSFORMERS! 🤖
# Self-attention, multiple layers, billions of parameters
# But the SAME gradient descent process!
```

**The training loop stays the same:**
1. Forward pass (make predictions)
2. Compute loss (how wrong are we?)
3. Backward pass (compute gradients)
4. Update weights (improve)

#### 🔹 **From Bigrams to Language Understanding**

The progression from what we just did to ChatGPT:

```
Bigram Model (us today)
   → Uses 1 previous character
   → 27×27 = 729 parameters
   → Loss: ~2.45
   ↓
MLP with Context
   → Uses multiple previous characters
   → Thousands of parameters
   → Better loss!
   ↓
Recurrent Networks (RNN, LSTM)
   → Can process sequences of any length
   → Millions of parameters
   ↓
Transformers (GPT, BERT, etc.)
   → Self-attention mechanism
   → Can attend to any part of context
   → Billions of parameters
   → Can write essays, code, and more!
```

**All using the same gradient descent optimization we just learned!**

#### 🔹 **Why Gradient Descent Wins**

| Aspect | Count-Based | Gradient-Based |
|--------|-------------|----------------|
| **Scalability** | Only works for simple models | Scales to billions of parameters |
| **Flexibility** | Hard to add complexity | Easy to add layers, attention, etc. |
| **Context** | Fixed context window | Can learn optimal context usage |
| **Features** | Manual feature engineering | Learns features automatically |
| **Limits** | Bigrams, maybe trigrams | Entire paragraphs, books! |

---

### So Cool! 🎉

**We arrived at the same result through completely different paths:**

1. **The Old Way (Count & Normalize):**
   - "Let me count how many times 'e' follows '.'"
   - "Let me normalize to get probabilities"
   - ✅ Works perfectly for simple bigrams
   - ❌ Doesn't scale beyond simple patterns

2. **The New Way (Gradient Descent):**
   - "Let me randomly initialize weights"
   - "Let me iteratively improve them by computing gradients"
   - ✅ Gets the same answer for bigrams
   - ✅ **Scales to arbitrarily complex models!**

**The magic:** We can now take this exact same training recipe (forward pass → loss → backward pass → update) and apply it to:
- Deeper networks
- More context characters
- Attention mechanisms
- **Transformers** (the architecture behind GPT, Claude, etc.)

**The journey ahead:**
```
You are here ⬇️
[Bigram Model] → [MLP] → [RNN] → [LSTM] → [Attention] → [Transformer] → [GPT-4]
     ✓
   2.45 loss
Same training process throughout! →  →  →  →  →  →
```

This is the foundation of modern AI. Same principles, increasingly sophisticated architectures. The gradient descent algorithm we just implemented scales from our simple 729-parameter bigram model all the way to trillion-parameter language models!

**That's the power of differentiable computing!** 💪

---

## Sampling from the Trained Model: Generating New Names

Now that we've trained the neural network, let's use it to generate new names! This shows the practical difference between the count-based and gradient-based approaches.

### The Code: Two Approaches to Sampling

```python
# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    
    out = []
    ix = 0  # Start with '.' (beginning token)
    while True:
        
        # __________
        # BEFORE: (Count-based approach)
        p = P[ix]
        # __________
        # NOW: (Gradient-based approach)
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdim=True) # probabilities for next character
        # __________
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # Stop at end token
            break
    print(''.join(out))
```

---

### Understanding the Sampling Process

#### **BEFORE: Count-Based Sampling**

```python
p = P[ix]
```

**What happens:**
- `P` is the pre-computed probability matrix from counts: `P = (N+1).float() / sum`
- Simply **look up** the row corresponding to current character `ix`
- Get probability distribution over all 27 next characters
- **One array lookup** - extremely fast!

**Characteristics:**
- ✅ Very fast (direct lookup)
- ✅ Simple and straightforward
- ❌ Static - computed once, never changes
- ❌ Limited to what we can pre-compute

---

#### **NOW: Gradient-Based Sampling**

```python
xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
p = counts / counts.sum(1, keepdim=True)
```

**What happens - step by step:**

1. **One-hot encode current character**
   ```python
   xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
   # If ix=0 ('.'): [1, 0, 0, ..., 0]
   # If ix=5 ('e'): [0, 0, 0, 0, 0, 1, 0, ..., 0]
   ```

2. **Apply learned weights (forward pass)**
   ```python
   logits = xenc @ W
   # Matrix multiply: (1×27) @ (27×27) = (1×27)
   # Extracts the row from W corresponding to current character
   ```

3. **Convert to counts**
   ```python
   counts = logits.exp()
   # Exponentiate to get positive "counts"
   ```

4. **Normalize to probabilities**
   ```python
   p = counts / counts.sum(1, keepdim=True)
   # Normalize so probabilities sum to 1.0
   ```

**Characteristics:**
- ❌ Slightly slower (needs forward pass computation)
- ✅ **Flexible** - can extend to complex architectures
- ✅ **Dynamic** - can condition on more context
- ✅ **Scalable** - same process works for deep networks

---

### Key Insight: They're Equivalent for Bigrams!

**Both approaches produce the same probability distribution:**

```python
# Count-based:
p_count = P[ix]  # Direct lookup from pre-computed matrix

# Neural network:
p_neural = softmax(W[ix])  # Forward pass through learned weights

# After training converges:
p_count ≈ p_neural  # Nearly identical!
```

**Why?** The neural network learned to encode the same information as the count matrix in its weights!

**Verification:**
```python
# The learned weight matrix W should be similar to log(P)
# (with some differences due to regularization)
torch.allclose(W, torch.log(P), atol=0.1)  # Approximately true!
```

---

### The Sampling Loop Breakdown

Let's trace through generating one name step-by-step:

```python
ix = 0  # Start: '.'
out = []

# Iteration 1:
p = get_probabilities(ix=0)  # Probabilities for first character
ix = sample_from(p)  # Say we get 'm' (ix=13)
out = ['m']

# Iteration 2:
p = get_probabilities(ix=13)  # Probabilities after 'm'
ix = sample_from(p)  # Say we get 'o' (ix=15)
out = ['m', 'o']

# Iteration 3:
p = get_probabilities(ix=15)  # Probabilities after 'o'
ix = sample_from(p)  # Say we get 'r' (ix=18)
out = ['m', 'o', 'r']

# Iteration 4:
p = get_probabilities(ix=18)  # Probabilities after 'r'
ix = sample_from(p)  # Say we get '.' (ix=0) - END!
out = ['m', 'o', 'r']

# Output: "mor"
```

---

### Why Use the Gradient-Based Approach for Sampling?

**If they produce the same results, why bother with the neural network approach?**

Great question! For bigrams, the count-based approach is actually better (simpler, faster). But the neural network approach shows its power when we extend beyond bigrams:

#### **Scaling to Complex Models:**

```python
# Bigram (what we have):
p = softmax(W[ix])  # One character context

# ⬇️ Easy to extend...

# Trigram (2-character context):
context = [prev_prev_char, prev_char]
xenc = encode(context)
p = softmax(xenc @ W)

# ⬇️ Keep extending...

# Multi-layer network (longer context):
h1 = relu(xenc @ W1)
h2 = relu(h1 @ W2)
p = softmax(h2 @ W3)

# ⬇️ All the way to...

# Transformer (entire sequence context):
attention_output = transformer(entire_sequence)
p = softmax(attention_output @ W_output)
```

**The sampling code structure stays nearly identical!** Just replace the probability computation with a more complex forward pass.

---

### Practical Comparison

| Aspect | Count-Based | Neural Network |
|--------|-------------|----------------|
| **Speed** | Instant lookup | Requires forward pass |
| **Memory** | Store full P matrix | Store weight matrix W |
| **Sampling quality** | Identical (for bigrams) | Identical (for bigrams) |
| **Extensibility** | Hard to scale | Easy to extend |
| **During training** | Static (recompute all) | Gradual improvement |
| **Inference** | Simple array access | Forward pass computation |

---

### Sample Output

Both approaches might generate names like:
```
mor.
axxyelle.
kinda.
jareigh.
gradelyn.
```

**The beauty:** These names don't exist in the training data, but they *sound* plausible because they follow the learned bigram patterns!

---

### The Bottom Line

**For this simple bigram model:**
- Count-based approach: Better for production (faster, simpler)
- Neural network approach: Better for learning and extending

**For complex models (transformers, etc.):**
- Count-based approach: Impossible (can't enumerate all contexts)
- Neural network approach: **Only viable option!**

The neural network framework may seem like overkill for bigrams, but it's the **foundation that scales** to GPT-4 and beyond. Same sampling loop, same forward pass structure, just deeper and more sophisticated! 🚀

---
