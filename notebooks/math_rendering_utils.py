# Enhanced Math Rendering Utilities for Jupyter Notebooks
# Add this cell at the beginning of your notebooks for beautiful LaTeX rendering

from IPython.display import display, Markdown, HTML, Math, Latex
import re

class MathRenderer:
    """Enhanced math rendering for beautiful LaTeX display in Jupyter notebooks."""

    @staticmethod
    def render_latex(text, display_mode=False):
        """
        Render text with proper LaTeX formatting.

        Args:
            text: Text containing LaTeX expressions
            display_mode: If True, render in display mode (centered, larger)

        Returns:
            IPython display object
        """
        if display_mode:
            return Latex(text)
        else:
            return Markdown(text)

    @staticmethod
    def normalize_latex_delimiters(text):
        """
        Convert various LaTeX delimiter styles to Jupyter-compatible format.

        Handles:
        - \\[ ... \\] → $$ ... $$ (display math)
        - \\( ... \\) → $ ... $ (inline math)
        - Already properly formatted delimiters
        """
        # Convert display math: \[ \] to $$ $$
        text = re.sub(r'\\\\\[', '$$', text)
        text = re.sub(r'\\\\\]', '$$', text)

        # Convert inline math: \( \) to $ $
        text = re.sub(r'\\\\\(', '$', text)
        text = re.sub(r'\\\\\)', '$', text)

        # Handle escaped backslashes in LaTeX  commands
        # Preserve actual LaTeX commands like \frac, \sum, etc.
        return text

    @staticmethod
    def render_math_text(text, max_length=None):
        """
        Render text with beautiful inline and display math.

        Args:
            text: Text containing math expressions
            max_length: Optional truncation length

        Returns:
            Rendered Markdown with proper LaTeX
        """
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."

        text = MathRenderer.normalize_latex_delimiters(text)
        return Markdown(text)

    @staticmethod
    def render_math_expression(expression):
        """
        Render a pure math expression (no surrounding text).

        Args:
            expression: LaTeX math expression

        Returns:
            Rendered Math object
        """
        # Remove delimiters if present
        expression = expression.strip()
        for delim in ['$$', '$', '\\[', '\\]', '\\(', '\\)']:
            expression = expression.replace(delim, '')

        return Math(expression)

    @staticmethod
    def render_question_answer(question, answer, show_answer=True):
        """
        Render a math question and answer beautifully.

        Args:
            question: Question text with LaTeX
            answer: Answer (may contain LaTeX)
            show_answer: Whether to display the answer

        Returns:
            Displays formatted question and answer
        """
        print("=" * 80)
        print("📝 PROBLEM")
        print("=" * 80)
        display(MathRenderer.render_math_text(question))

        if show_answer:
            print("\n" + "─" * 80)
            print("✓ ANSWER")
            print("─" * 80)

            # Check if answer contains LaTeX
            if any(delim in str(answer) for delim in ['$', '\\\\', 'frac', 'sqrt', 'sum', 'int']):
                display(MathRenderer.render_math_text(str(answer)))
            else:
                # Simple answer - display as code
                display(HTML(f'<code style="font-size: 14px; background: #f0f0f0; padding: 5px;">{answer}</code>'))
            print()

    @staticmethod
    def render_comparison_table(data_rows, title="Comparison"):
        """
        Render a beautiful comparison table with proper formatting.

        Args:
            data_rows: List of dictionaries with comparison data
            title: Table title

        Returns:
            Displays HTML table
        """
        html = f"""
        <div style="margin: 20px 0;">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                {title}
            </h3>
            <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
                <thead>
                    <tr style="background-color: #3498db; color: white;">
        """

        # Add headers
        if data_rows:
            for key in data_rows[0].keys():
                html += f'<th style="padding: 12px; text-align: left; border: 1px solid #ddd;">{key}</th>'

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        # Add rows with alternating colors
        for i, row in enumerate(data_rows):
            bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
            html += f'<tr style="background-color: {bg_color};">'
            for value in row.values():
                # Check if value contains LaTeX
                if isinstance(value, str) and any(delim in value for delim in ['$', '\\\\']):
                    # Render inline math
                    value = MathRenderer.normalize_latex_delimiters(value)
                html += f'<td style="padding: 10px; border: 1px solid #ddd;">{value}</td>'
            html += '</tr>'

        html += """
                </tbody>
            </table>
        </div>
        """

        display(HTML(html))

    @staticmethod
    def render_reasoning_steps(completion, title="Reasoning Process"):
        """
        Render reasoning steps with proper math formatting.

        Args:
            completion: Generated completion text
            title: Section title
        """
        print("\n" + "=" * 80)
        print(f"🧠 {title.upper()}")
        print("=" * 80)

        # Split by common step indicators
        steps = re.split(r'\n(?=Step \d+|Therefore|Thus|Hence|Finally)', completion)

        for i, step in enumerate(steps, 1):
            if step.strip():
                print(f"\n[Step {i}]")
                print("─" * 80)
                display(MathRenderer.render_math_text(step.strip()))
        print()


# Convenience functions for quick use
def show_math(text):
    """Quick display of math text."""
    return MathRenderer.render_math_text(text)

def show_equation(latex_expr):
    """Quick display of a math equation."""
    return MathRenderer.render_math_expression(latex_expr)

def show_question(question, answer=None):
    """Quick display of question and answer."""
    MathRenderer.render_question_answer(question, answer if answer else "")

# Test examples (comment out in production)
if __name__ == "__main__":
    # Example 1: Simple inline math
    print("Example 1: Inline Math")
    display(show_math("The solution is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$"))

    # Example 2: Display math
    print("\nExample 2: Display Math")
    display(show_math("$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$"))

    # Example 3: Question and answer
    print("\nExample 3: Question with Answer")
    show_question(
        "Find the value of $\\sum_{k=1}^{n} k^2$",
        "$\\frac{n(n+1)(2n+1)}{6}$"
    )

    print("\n✓ Math rendering utilities loaded successfully!")
