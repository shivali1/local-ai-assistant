from ollama import chat
from pathlib import Path
import json

chat_dir = Path("Chats")
chat_dir.mkdir(exist_ok=True)

print("Choose an assistant: 1. DSA Interviewer 2. AI Mentor 3. Resume Reviewer 4. Travel Planner")
num = int(input("Enter your choice: "))

prompts = {
    1: """
    You are a DSA Interviewer.

    Rules:
    - Introduce yourself and your role.
    - Explain concepts clearly.
    - Give examples when necessary.
    - Keep answers concise.
    """,

    2: """
    You are an experienced AI Engineer and mentor.

    Rules:
    - Explain concepts clearly.
    - Give examples when necessary.
    - Keep answers only in bullets.
    """,

    3: """
    You are a Resume Reviewer.

    Rules:
    - Introduce yourself and your role.
    - Explain concepts clearly.
    - Give examples when necessary.
    - Keep answers concise.
    """,

    4: """
    You are a Travel Planner.

    Rules:
    - Introduce yourself and your role.
    - Explain concepts clearly.
    - Give examples when necessary.
    - Keep answers concise.
    """
}

messages = []
title = "Untitled"

files = {
    1: "dsa",
    2: "ai",
    3: "resume",
    4: "travel"
}

prefix = files[num]

existing_chats = sorted(chat_dir.glob(f"{prefix}*.json"))

if len(existing_chats) == 0:

    print("\nNo previous chats found. Starting a new chat.\n")

    filename = chat_dir / f"{prefix}1.json"

    messages = [
        {
            "role": "system",
            "content": prompts[num]
        }
    ]

else:

    print("\nPrevious Chats:\n")

    for i, file in enumerate(existing_chats, start=1):

        with open(file, "r") as f:
            chat_data = json.load(f)

        print(f"{i}. {chat_data['title']}")

    print(f"{len(existing_chats)+1}. Start New Chat")

    choice = int(input("\nChoose: "))

    if choice <= len(existing_chats):

        filename = existing_chats[choice-1]

        with open(filename, "r") as f:
            chat_data = json.load(f)

        title = chat_data["title"]
        messages = chat_data["messages"]

        print("\nPrevious chat loaded.\n")

    else:

        filename = chat_dir / f"{prefix}{len(existing_chats)+1}.json"

        messages = [
            {
                "role": "system",
                "content": prompts[num]
            }
        ]

        print("\nNew chat started.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":

        # -------- Generate Chat Title --------

        title_prompt = f"""
Generate a concise title (maximum 5 words) for the following conversation.

Return ONLY the title.

Conversation:

{messages}
"""

        response = chat(
            model="qwen2:7b",
            messages=[
                {
                    "role": "user",
                    "content": title_prompt
                }
            ]
        )

        title = response["message"]["content"].strip()

        # -------- Save --------

        chat_data = {
            "title": title,
            "messages": messages
        }

        with open(filename, "w") as f:
            json.dump(chat_data, f, indent=4)

        print(f"\nChat saved as: {title}")
        break

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    stream = chat(
        model="qwen2:7b",
        messages=messages,
        stream=True
    )

    answer = ""

    for chunk in stream:

        content = chunk["message"]["content"]

        print(content, end="", flush=True)

        answer += content

    print()

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    chat_data = {
        "title": title,
        "messages": messages
    }

    with open(filename, "w") as f:
        json.dump(chat_data, f, indent=4)