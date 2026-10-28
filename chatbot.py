import ollama
import csv
import os
import logging
from datetime import datetime

MODEL_NAME = "gemma3:1b"

os.makedirs("dataset", exist_ok=True)

CSV_FILE = "dataset/chat_history.csv"
RESPONSES_FILE = "dataset/responses.csv"
QUESTIONS_FILE = "dataset/sample_questions.csv"
LOG_FILE = "dataset/chatbot_logs.txt"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True
)

SYSTEM_PROMPT = (
    "You are an Education Assistant chatbot. "
    "Answer in simple English. Explain concepts step by step "
    "and provide examples when useful. For coding questions, "
    "give beginner-friendly explanations and code. "
    "If unsure, say so clearly."
)

CHAT_HEADERS = [
    "timestamp",
    "user_message",
    "chatbot_response"
]

RESPONSE_HEADERS = [
    "timestamp",
    "question_id",
    "category",
    "question",
    "model",
    "response",
    "response_length_words"
]


def create_csv_if_needed(filename, headers):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(headers)


create_csv_if_needed(CSV_FILE, CHAT_HEADERS)
create_csv_if_needed(RESPONSES_FILE, RESPONSE_HEADERS)


def ask_chatbot(question, history=None, system_prompt=None):
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": question
    })

    result = ollama.chat(
        model=MODEL_NAME,
        messages=messages
    )

    return result["message"]["content"]


def load_sample_questions():
    if not os.path.exists(QUESTIONS_FILE):
        print("Error: dataset/sample_questions.csv was not found.")
        logging.error("Sample questions CSV not found")
        return []

    with open(
        QUESTIONS_FILE, "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def get_completed_ids():
    completed_ids = set()

    with open(
        RESPONSES_FILE, "r", newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            question_id = row.get("question_id", "").strip()

            if question_id:
                completed_ids.add(question_id)

    return completed_ids


def save_response(question_id, category, question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(
        RESPONSES_FILE, "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            question_id,
            category,
            question,
            MODEL_NAME,
            answer,
            len(answer.split())
        ])


def run_sample_questions():
    questions = load_sample_questions()

    if not questions:
        print("No sample questions found.")
        return

    completed_ids = get_completed_ids()

    pending = [
        row for row in questions
        if row.get("question_id", "").strip() not in completed_ids
    ]

    if not pending:
        print("All sample questions already have saved responses.")
        print("Existing results were preserved.")
        return

    print(f"\nGenerating responses for {len(pending)} questions...")
    logging.info("Sample test started")

    for row in pending:
        question_id = row.get("question_id", "").strip()
        category = row.get("category", "").strip()
        question = row.get("question", "").strip()

        if not question:
            continue

        try:
            answer = ask_chatbot(question)

            save_response(
                question_id, category, question, answer
            )

            completed_ids.add(question_id)

            print(f"Saved response for question {question_id}")
            logging.info(
                "Saved sample response for question ID %s",
                question_id
            )

        except Exception:
            logging.exception(
                "Failed on question ID %s", question_id
            )
            print(
                f"Question {question_id} failed. "
                "Check dataset/chatbot_logs.txt."
            )

    print("\nSample question processing finished.")
    print("Check dataset/responses.csv.")


def run_prompt_comparison():
    questions = load_sample_questions()

    if not questions:
        print("No sample questions found.")
        return

    original_prompt = (
        "You are an Education Assistant. "
        "Answer the student's question in simple English."
    )

    improved_prompt = (
        "You are an expert Education Assistant for students. "
        "Answer accurately in simple English. Explain the main "
        "idea step by step and give a relevant example when useful. "
        "For coding questions, explain the logic and provide "
        "beginner-friendly code. Organize the answer clearly. "
        "Do not invent facts. If you are unsure, state that clearly."
    )

    completed_ids = get_completed_ids()

    total_saved = 0

    print("\nStarting Original vs Improved Prompt comparison...")
    print("Actual responses will be saved in dataset/responses.csv.\n")

    for row in questions:
        question_id = row.get("question_id", "").strip()
        category = row.get("category", "").strip()
        question = row.get("question", "").strip()

        if not question_id or not question:
            continue

        prompt_versions = [
            ("original", original_prompt),
            ("improved", improved_prompt)
        ]

        for version, prompt in prompt_versions:
            result_id = f"{question_id}-{version}"

            if result_id in completed_ids:
                print(f"Already saved: {result_id}")
                continue

            try:
                answer = ask_chatbot(
                    question,
                    system_prompt=prompt
                )

                save_response(
                    result_id,
                    category,
                    f"[{version.upper()} PROMPT] {question}",
                    answer
                )

                completed_ids.add(result_id)
                total_saved += 1

                print(f"Saved comparison: {result_id}")

                logging.info(
                    "Prompt comparison saved: %s",
                    result_id
                )

            except Exception:
                logging.exception(
                    "Prompt comparison failed: %s",
                    result_id
                )
                print(
                    f"{result_id} failed. "
                    "Check dataset/chatbot_logs.txt."
                )

    print(f"\nComparison finished. New responses saved: {total_saved}")
    print("Open dataset/responses.csv to compare the answers.")


def run_interactive_chat():
    history = []

    print("\nWelcome to the Education AI Chatbot!")
    print("Type 'exit' to stop the conversation.\n")

    logging.info(
        "Interactive chatbot started using model %s",
        MODEL_NAME
    )

    while True:
        try:
            user_input = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Goodbye!")
            break

        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        if not user_input:
            print("Chatbot: Please enter a question.")
            continue

        try:
            bot_reply = ask_chatbot(user_input, history)

            print("Chatbot:", bot_reply)

            history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": bot_reply}
            ])

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(
                CSV_FILE, "a", newline="", encoding="utf-8"
            ) as file:
                csv.writer(file).writerow([
                    timestamp,
                    user_input,
                    bot_reply
                ])

            logging.info("Interactive conversation saved")

        except Exception:
            logging.exception(
                "Failed to process interactive question"
            )
            print(
                "Chatbot error occurred. "
                "Check dataset/chatbot_logs.txt."
            )


if __name__ == "__main__":
    print("\n===== Education AI Chatbot =====")
    print("1. Chat with the Education AI Chatbot")
    print("2. Generate responses for sample questions")
    print("3. Compare Original and Improved Prompts")

    choice = input("Choose 1, 2, or 3: ").strip()

    if choice == "1":
        run_interactive_chat()

    elif choice == "2":
        run_sample_questions()

    elif choice == "3":
        run_prompt_comparison()

    else:
        print("Invalid choice. Run the program again.")