import json
import re
import textwrap


class styles:
    HEADER = '\033[95m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def print_md(text):
    """
    Translates simple markdown and wraps text for better terminal display.
    """
    text = re.sub(r'\*\*(.*?)\*\*', f'{styles.BOLD}\\1{styles.ENDC}', text)
    paragraphs = text.split('\n')
    wrapped_paragraphs = [textwrap.fill(p, width=90) for p in paragraphs]
    final_text = '\n'.join(wrapped_paragraphs)
    print(final_text)


def display_report_fn(data):
    """
    Loads, parses, and prints the formatted IELTS report from a JSON file.
    """

    print("\n" + "=" * 80)
    print(f"{styles.HEADER}{styles.BOLD}📝 IELTS WRITING EVALUATION REPORT{styles.ENDC}")
    print("=" * 80)

    if "original_question" in data:
        print(f"\n{styles.HEADER}{styles.UNDERLINE}Original Question:{styles.ENDC}")
        print_md(data["original_question"])

    if "student_essay" in data:
        print(f"\n{styles.HEADER}{styles.UNDERLINE}Student's Essay:{styles.ENDC}")
        print_md(data["student_essay"])
        print("-" * 80)

    sections = {
        "task_response": "✅ Task Response",
        "coherence_and_cohesion": "🔗 Coherence and Cohesion",
        "lexical_resource": "📚 Lexical Resource",
        "grammatical_range_and_accuracy": "📝 Grammatical Range and Accuracy"
    }

    for key, title in sections.items():
        if key in data:
            print(f"\n\n{styles.GREEN}{styles.BOLD}{styles.UNDERLINE}--- {title} ---{styles.ENDC}\n")
            print_md(data[key])

    if "aggregated_result" in data:
        print(
            f"\n\n{styles.WARNING}{styles.BOLD}{styles.UNDERLINE}--- 📊 Overall Summary & Final Score ---{styles.ENDC}\n")
        print_md(data["aggregated_result"])

    print("\n" + "=" * 80 + "\n")


