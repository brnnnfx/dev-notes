Exercise - Research project

You're using AI to help analyze a large dataset for a research paper.

- **Delegation:** How would you divide the analytical work between yourself and the AI?
	- I would perform all final validation of completeness, accuracy and readiness
	- I would provide AI with examples of things to avoid and watch out for (PI, SPII etc.)
	- I would converse with AI as a partner to decide for the particular data set the best practices to use (how to model, what analysis to perform based on the data, what means to present the analytical results) but I make the final decision
	- I would give AI instructions of how to cleanse the data, AI does the cleansing, I validate the results
- **Description:** What context would the AI need to understand about your research question to do its share of the tasks well?
	- What is in scope / out of scope, including SPII
	- Provide clear instructions file that follow established best practices
	- Explicitly state what not to do
- **Discernment:** How would you verify the AI's analysis for accuracy?
	- Use proven statistical models and / or external tools to validate accuracy
	- Human eye / smell test
	- Ask an SME to validate once I've reviewed
- **Diligence:** What ethical considerations might arise when publishing AI-assisted research?
	- Plagiarism
	- Over confidence in hallucinations or partial truths may result in future papers that cite it being based on incorrect information

Exercise - Discuss something I am already familiar with
⏲️ Time-boxed to 5 minutes
🤖 Model = mistral-small-2506

I am keeping a record of this conversation to observe how my prompting improves with more practice.

Instructions
```
Choose a topic you know well and enjoy discussing such as a hobby, professional interest, favorite book series, etc.
Have a natural conversation with Claude about this topic, like you would with someone who shares your interest.
Try to notice moments where:

    AI enhances your thinking
    You need to clarify or correct the AI's understanding
    Your expertise leads you to evaluate the AI's responses
```

💬 Brennan
```
I am trying to learn the capabilities of generative AI. As part of an exercise I want to have a conversation about something I know a lot about.

Let’s talk about being a parent of a young child in kidergarten.
```
💬 AI
```
That sounds like a great way to explore generative AI’s capabilities—by grounding the conversation in your real-life experience! I’d love to learn from you about what it’s like to be a parent of a young child in kindergarten. What’s something you’ve noticed or learned recently that you’d like to share or discuss?
```
💬 Brennan
```
I am amazed at how present and in the moment they are. The concerns of tomorrow do not worry them; I want a bit of that.
```
💬 AI
```
That’s such a beautiful observation—and something many parents secretly envy! Young children really do live in the moment, and it’s a great reminder of how much we let future worries pull us away from the present. Do you find yourself trying to mimic that kind of presence with your child, or does it just make you reflect on your own habits?
```
💬 Brennan
```
I definitely reflect on my own habits for sure, but I’d like to set an intention to try to mimic being ‘child like’ myself. I’m not sure how to do that practically.

What do you do - or since your an LLM - what would you suggest I do to get started?
```
💬 AI
```
That’s a wonderful intention! One practical way to start is to set aside small moments in your day where you deliberately focus only on what’s happening right now—just like a child would. For example, when you’re with your child, try to fully engage in whatever they’re doing, whether it’s playing, talking, or even just walking together. Put your phone away, avoid multitasking, and let yourself get absorbed in the moment with them. Over time, these small practices can help you recapture that sense of presence.

Another idea is to find activities that naturally pull you into the present, like drawing, dancing, or even just sitting quietly and observing your surroundings. Children often get lost in these kinds of activities without overthinking them. What do you think might work best for you based on your daily routine?
```

💬 Brennan
```
I like the idea of taking 5 or so minutes to draw or write a bit. I have a ‘worry journal’ I use to colour worries away but I don’t use it as much as I like (the worries build up).

I can get stuck when I stare at a ‘blank page’. What do you do to get started doing creative drawing?
```

💬 AI
```
That’s a great approach—using drawing or writing as a way to ground yourself in the moment. When you’re staring at a blank page, sometimes the trick is to just start with something simple and imperfect. You could begin by doodling shapes, scribbling lines, or even tracing something random with your eyes closed. The goal isn’t to create a masterpiece; it’s to engage your hands and mind in a low-pressure way.

Another idea is to use prompts or constraints to spark creativity. For example, you could set a timer for 5 minutes and draw whatever comes to mind without lifting your pen, or write a single sentence about whatever you’re feeling in that moment. Over time, these small starts can help you build a habit and reduce the pressure to make everything “perfect.”

Have you tried any of these techniques before, or is there a specific type of drawing or writing that feels most appealing to you right now?
```

Exercise - Learn something new
⏲️ Time-boxed to 5 minutes
🤖 Model = mistral-small-2506

💬 Brennan
```
I am trying to learn the capabilities of generative AI. As part of an exercise I want to have a conversation about something I know very little about; the statistical principle of empirical risk minimization.

Give me a gentle introduction to the topic using an example that I might be familiar from things I might encounter in my family life at home. Use simple terms and only use technical terms when it is absolutely necessary.
```
💬 AI
```
Imagine you’re trying to teach a very young child to recognize the difference between apples and oranges. You show them a bunch of pictures and say, “This is an apple,” or “This is an orange.” After a while, the child starts to notice patterns: apples are usually red or green, round, and have a stem; oranges are usually orange, round, and have a bumpy texture. The child is learning to make a rule: “If it’s red and round and has a stem, it’s an apple; if it’s orange and bumpy, it’s an orange.”

Now, suppose you want to test how well the child can tell apples from oranges. You show them a new set of pictures they’ve never seen before and ask them to label each one. The child’s accuracy on this new set is a measure of how well they’ve learned the rule. If the child does well, it means their rule is good at generalizing to new examples.

**Empirical Risk Minimization (ERM)** is a fancy way of saying: “Find the rule that makes the fewest mistakes on the examples you’ve already seen.” In the child’s case, the rule is “red, round, stem = apple” and “orange, bumpy = orange.” The child’s mistakes on the training pictures (the ones you showed them first) are the “empirical risk.” ERM is about picking the rule that gives the smallest number of mistakes on those training pictures.

In real life, this idea is used in things like spam filters (teaching a computer to recognize spam emails by showing it lots of examples of spam and not-spam, then picking the rule that makes the fewest mistakes on those examples) or recommendation systems (teaching a computer to recommend movies you might like by showing it lots of examples of movies you’ve liked and disliked in the past, then picking the rule that makes the fewest mistakes on those examples).

So, in simple terms:
- **Empirical Risk** = the mistakes you make on the examples you’ve already seen.
- **Minimization** = trying to make those mistakes as small as possible.
- **Empirical Risk Minimization** = find the rule that makes the fewest mistakes on the examples you’ve already seen.
```

💬 Brennan
```
Thanks. In what workflow or process is this rule applied? And where in the flow?
```

💬 AI
```
Great question! Let’s break down where **Empirical Risk Minimization (ERM)** fits into a typical machine learning workflow, using the same apple-orange example to keep it concrete.

---

### **Typical Machine Learning Workflow (Simplified)**

1. **Problem Definition**
   - *What are you trying to predict?*
   - Example: “I want to teach a computer to tell apples from oranges.”

2. **Data Collection**
   - *Gather examples (data) of apples and oranges.*
   - Example: Take 100 pictures of apples and 100 pictures of oranges.

3. **Feature Extraction**
   - *Turn raw data (pictures) into numbers the computer can understand.*
   - Example: For each picture, measure color (red/green/orange), roundness (1-10 scale), stem presence (yes/no), bumpy texture (yes/no).

4. **Model Selection**
   - *Choose a rule-making tool (algorithm).*
   - Example: A simple rule like “if red and round and stem, then apple; else orange.”

5. **Training (Where ERM Happens!)**
   - *Use the training data to find the best rule.*
   - Example: Show the computer all 200 pictures and its labels. The computer tries different rules and picks the one that makes the fewest mistakes on these pictures. This is **Empirical Risk Minimization**—minimizing mistakes on the training data.

6. **Evaluation**
   - *Test the rule on new, unseen data to see if it works.*
   - Example: Show the computer 50 new pictures it’s never seen before. Count how many mistakes it makes.

7. **Deployment**
   - *Use the rule in the real world.*
   - Example: Put the apple-orange classifier on your phone to help sort fruit at home.

---

### **Where ERM Fits In**
ERM is the **core process in step 5 (Training)**. It’s the method the computer uses to find the best rule by minimizing mistakes on the training data. Without ERM, the computer wouldn’t know which rule to pick—it would just guess randomly.

---

### **Real-World Analogy**
Think of ERM like a chef perfecting a recipe. The chef tries different ingredient combinations (rules) and picks the one that tastes best (makes the fewest mistakes) when tested on dishes they’ve already made (training data). Only after perfecting the recipe do they serve it to guests (evaluation and deployment).

---

### **Key Takeaway**
ERM is the engine that drives the learning process in machine learning. It’s not a separate step, but the method used during training to find the best rule by minimizing mistakes on the data you already have.
```

💬 Brennan
```
Give me a code snippet example that illustrates how this would look like. Explain each function / operator etc. in plain language so I undestand the purpose and rationale for everything.
```

💬 AI
```
Here’s a simple Python code snippet that demonstrates **Empirical Risk Minimization (ERM)** using a classic example: teaching a computer to distinguish between apples and oranges based on color and texture. I’ll use a very simple model (a decision tree) and explain every part in plain language.

---

### **Code Example: Apple vs. Orange Classifier**
```python
# Step 1: Import the tools we need
from sklearn.tree import DecisionTreeClassifier  # A simple rule-making tool
from sklearn.metrics import accuracy_score       # A way to measure mistakes

# Step 2: Create some training data (our "family life" example)
# Each row is a fruit: [color (0=red, 1=orange), texture (0=smooth, 1=bumpy)]
# Each label is 0 for apple, 1 for orange
X_train = [
    [0, 0],  # red, smooth (apple)
    [0, 0],  # red, smooth (apple)
    [0, 1],  # red, bumpy (apple)
    [1, 0],  # orange, smooth (orange)
    [1, 1],  # orange, bumpy (orange)
    [1, 1],  # orange, bumpy (orange)
]
y_train = [0, 0, 0, 1, 1, 1]  # 0=apple, 1=orange

# Step 3: Create the model (our rule-making tool)
model = DecisionTreeClassifier()

# Step 4: Train the model (this is where ERM happens!)
# The model looks at X_train and y_train, then finds the best rule
# to minimize mistakes on this training data.
model.fit(X_train, y_train)

# Step 5: Test the model on new data (evaluation)
X_test = [
    [0, 0],  # red, smooth (should be apple)
    [1, 1],  # orange, bumpy (should be orange)
    [0, 1],  # red, bumpy (should be apple)
]
y_test = [0, 1, 0]  # Expected labels for the test data

# Step 6: Make predictions and check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)  # What the model guessed
print("Accuracy:", accuracy)        # How many mistakes it made

---

### **Explanation of Each Part**

---

#### **1. Importing Tools**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```
- **`DecisionTreeClassifier`**: This is a simple tool that creates rules like “if color is red and texture is smooth, then it’s an apple.” It’s called a “decision tree” because it makes decisions (rules) in a tree-like structure.
- **`accuracy_score`**: This is a way to measure how many mistakes the model made. It compares the model’s guesses to the correct answers and tells you the percentage of correct guesses.

---

#### **2. Training Data (`X_train` and `y_train`)**
```python
X_train = [
    [0, 0],  # red, smooth (apple)
    [0, 0],  # red, smooth (apple)
    [0, 1],  # red, bumpy (apple)
    [1, 1],  # orange, bumpy (orange)
    [1, 0],  # orange, smooth (orange)
    [1, 1],  # orange, bumpy (orange)
]
y_train = [0, 0, 0, 1, 1, 1]  # 0=apple, 1=orange

- **`X_train`**: This is a list of examples. Each example is a list of numbers representing features (characteristics) of the fruit:
  - `0` = red, `1` = orange (color)
  - `0` = smooth, `1` = bumpy (texture)
- **`y_train`**: This is a list of labels (answers) for each example. `0` means “apple,” and `1` means “orange.”

---

#### **3. Creating the Model**
```python
model = DecisionTreeClassifier()
```
- This line creates a new, empty decision tree model. It’s like a blank recipe book waiting to be filled with rules.

---

#### **4. Training the Model (ERM in Action!)**
```python
model.fit(X_train, y_train)
```
- **`fit`**: This is the command that trains the model. It looks at all the training data (`X_train` and `y_train`) and tries to find the best rules to minimize mistakes.
- **ERM in action**: The model is trying different rules (e.g., “if color is red, then apple; if color is orange, then orange”) and picking the one that makes the fewest mistakes on the training data. This is **Empirical Risk Minimization**—minimizing mistakes on the data you already have.

---

#### **5. Testing the Model**
```python
X_test = [
    [0, 0],  # red, smooth (should be apple)
    [1, 1],  # orange, bumpy (should be orange)
    [0, 1],  # red, bumpy (should be apple)
]
y_test = [0, 1, 0]  # Expected labels for the test data
```
- **`X_test`**: New examples the model hasn’t seen before. These are like the “test pictures” you show the child after teaching them.
- **`y_test`**: The correct answers for these new examples.

---

#### **6. Making Predictions and Checking Accuracy**
```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```
- **`predict`**: The model uses the rules it learned during training to guess the labels for the new examples in `X_test`.
- **`accuracy_score`**: This compares the model’s guesses (`predictions`) to the correct answers (`y_test`) and tells you how many mistakes it made. For example, if it got 2 out of 3 correct, the accuracy is `0.66` (66%).

---

### **What This Code Demonstrates**
- The model learns rules from the training data (`X_train`, `y_train`).
- The learning process (via `fit`) is where **Empirical Risk Minimization** happens: the model finds the rules that minimize mistakes on the training data.
- The model is then tested on new data (`X_test`, `y_test`) to see how well it generalizes.

---

### **Key Takeaway**
This code is a tiny, simplified version of how ERM is used in real-world machine learning. In practice, the data is much larger and more complex, but the core idea is the same: **find the rules that make the fewest mistakes on the data you already have.**
```