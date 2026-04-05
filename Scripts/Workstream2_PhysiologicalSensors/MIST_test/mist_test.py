import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

LEVEL_DURATION_SEC = 120
QUESTION_DURATION_SEC = 10
AVERAGE_PERFORMANCE = 90.0
FEEDBACK_MS = 1200
WINDOW_W = 1180
WINDOW_H = 780

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'mist_logs'
LOG_DIR.mkdir(exist_ok=True)


@dataclass
class Question:
    display: str
    eval_expr: str
    answer: int


class MistApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('MIST-style Level 1-5 Test')
        self.root.geometry(f'{WINDOW_W}x{WINDOW_H}')
        self.root.configure(bg='white')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.participant_id = datetime.now().strftime('P%Y%m%d_%H%M%S')
        self.start_perf = None
        self.experiment_started = False

        self.current_level = 1
        self.level_end_perf = None
        self.question_deadline_perf = None
        self.level_tick_job = None
        self.question_tick_job = None
        self.next_question_job = None
        self.current_question = None
        self.current_question_start_perf = None
        self.selected_answer = None
        self.question_active = False
        self.feedback_active = False

        self.total_answered = 0
        self.total_correct = 0
        self.level_answered = 0
        self.level_correct = 0
        self.question_index_global = 0
        self.question_index_level = 0

        self.log_file = None
        self.log_writer = None
        self.log_path = None

        self._init_log()
        self._build_ui()
        self._show_intro()

    def _init_log(self):
        filename = LOG_DIR / f'mist_{self.participant_id}.csv'
        self.log_file = open(filename, 'w', newline='', encoding='utf-8')
        self.log_path = filename
        fields = [
            'participant_id', 'timestamp', 't_rel_s', 'level', 'event',
            'question_index_global', 'question_index_level', 'question_display',
            'selected_answer', 'correct_answer', 'is_correct', 'rt_ms',
            'level_time_left_s', 'question_time_left_s', 'accuracy_pct', 'note'
        ]
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=fields)
        self.log_writer.writeheader()
        self.log_file.flush()

    def _log(self, event, selected_answer='', is_correct='', rt_ms='', note=''):
        now = datetime.now().isoformat(timespec='milliseconds')
        t_rel_s = ''
        if self.start_perf is not None:
            t_rel_s = round(time.perf_counter() - self.start_perf, 3)

        level_left = ''
        if self.level_end_perf is not None:
            level_left = round(max(0.0, self.level_end_perf - time.perf_counter()), 3)

        q_left = ''
        if self.question_deadline_perf is not None and self.question_active:
            q_left = round(max(0.0, self.question_deadline_perf - time.perf_counter()), 3)

        accuracy = ''
        if self.total_answered > 0:
            accuracy = round(100.0 * self.total_correct / self.total_answered, 2)

        self.log_writer.writerow({
            'participant_id': self.participant_id,
            'timestamp': now,
            't_rel_s': t_rel_s,
            'level': self.current_level if self.experiment_started else '',
            'event': event,
            'question_index_global': self.question_index_global,
            'question_index_level': self.question_index_level,
            'question_display': self.current_question.display if self.current_question else '',
            'selected_answer': selected_answer,
            'correct_answer': self.current_question.answer if self.current_question else '',
            'is_correct': is_correct,
            'rt_ms': rt_ms,
            'level_time_left_s': level_left,
            'question_time_left_s': q_left,
            'accuracy_pct': accuracy,
            'note': note,
        })
        self.log_file.flush()

    def _build_ui(self):
        self.header = tk.Frame(self.root, bg='white')
        self.header.pack(fill='x', pady=(16, 6))

        self.title_label = tk.Label(
            self.header,
            text='MIST-style Arithmetic Test',
            font=('Arial', 28, 'bold'),
            bg='white'
        )
        self.title_label.pack()

        self.status_label = tk.Label(
            self.header,
            text='',
            font=('Arial', 18),
            bg='white'
        )
        self.status_label.pack(pady=(4, 0))

        self.timer_label = tk.Label(
            self.header,
            text='',
            font=('Arial', 16),
            bg='white'
        )
        self.timer_label.pack(pady=(4, 0))

        self.qtime_frame = tk.Frame(self.header, bg='white')
        self.qtime_frame.pack(pady=(6, 0))

        self.qtime_text_label = tk.Label(
            self.qtime_frame,
            text=f'Question time left: {QUESTION_DURATION_SEC:.1f}s',
            font=('Arial', 15),
            bg='white'
        )
        self.qtime_text_label.pack()

        self.qtime_canvas = tk.Canvas(
            self.qtime_frame,
            width=420,
            height=22,
            bg='white',
            highlightthickness=1,
            highlightbackground='black'
        )
        self.qtime_canvas.pack(pady=(4, 0))
        self.qtime_bar = self.qtime_canvas.create_rectangle(
            0, 0, 420, 22, fill='#4CAF50', width=0
        )

        self.center = tk.Frame(self.root, bg='white')
        self.center.pack(expand=True, fill='both')

        self.instruction_label = tk.Label(
            self.center,
            text='',
            font=('Arial', 20),
            bg='white',
            justify='center',
            wraplength=1000
        )
        self.instruction_label.pack(pady=(10, 20))

        self.question_label = tk.Label(
            self.center,
            text='',
            font=('Arial', 42, 'bold'),
            bg='white'
        )
        self.question_label.pack(pady=20)

        self.selected_label = tk.Label(
            self.center,
            text='Selected answer: None',
            font=('Arial', 20),
            bg='white'
        )
        self.selected_label.pack(pady=(0, 20))

        self.digit_frame = tk.Frame(self.center, bg='white')
        self.digit_buttons = []
        for n in range(10):
            btn = tk.Button(
                self.digit_frame,
                text=str(n),
                width=4,
                height=2,
                font=('Arial', 20),
                command=lambda x=n: self.select_answer(x)
            )
            r, c = divmod(n, 5)
            btn.grid(row=r, column=c, padx=8, pady=8)
            self.digit_buttons.append(btn)

        self.submit_btn = tk.Button(
            self.center,
            text='Submit',
            font=('Arial', 20, 'bold'),
            width=12,
            command=self.submit_answer
        )

        self.feedback_label = tk.Label(
            self.center,
            text='',
            font=('Arial', 20, 'bold'),
            bg='white',
            justify='center'
        )
        self.feedback_label.pack(pady=16)

        self.performance_label = tk.Label(
            self.center,
            text='',
            font=('Arial', 18),
            bg='white'
        )
        self.performance_label.pack(pady=(0, 18))

        self.start_btn = tk.Button(
            self.center,
            text='Start Test',
            font=('Arial', 22, 'bold'),
            width=14,
            command=self.start_test
        )
        self.start_btn.pack(pady=20)

    def _show_intro(self):
        self.status_label.config(
            text='Levels 1 to 5 will appear in order. Each level lasts 2 minutes.'
        )
        self.timer_label.config(text='Each question must be answered within 10 seconds.')
        self.instruction_label.config(
            text=(
                'Use the mouse to select one digit from 0 to 9, then click Submit.\n'
                'After each answer, feedback will show Correct / Incorrect and whether your performance is below or above average.'
            )
        )
        self.question_label.config(text='')
        self.selected_label.config(text='Selected answer: None')
        self.feedback_label.config(text='')
        self.performance_label.config(
            text=f'Average performance target: {AVERAGE_PERFORMANCE:.0f}%'
        )
        self._update_question_progress(QUESTION_DURATION_SEC)

    def _update_question_progress(self, seconds_left: float):
        total_w = 420
        ratio = max(0.0, min(1.0, seconds_left / QUESTION_DURATION_SEC))
        bar_w = total_w * ratio

        self.qtime_canvas.coords(self.qtime_bar, 0, 0, bar_w, 22)

        if ratio > 0.6:
            color = '#4CAF50'
        elif ratio > 0.3:
            color = '#FFC107'
        else:
            color = '#F44336'

        self.qtime_canvas.itemconfig(self.qtime_bar, fill=color)
        self.qtime_text_label.config(text=f'Question time left: {seconds_left:0.1f}s')

    def start_test(self):
        self.start_btn.pack_forget()
        self.digit_frame.pack(pady=8)
        self.submit_btn.pack(pady=(8, 14))
        self.start_perf = time.perf_counter()
        self.experiment_started = True
        self._log('EXPERIMENT_START')
        self.start_level(1)

    def start_level(self, level):
        self._cancel_pending()
        self.current_level = level
        self.level_answered = 0
        self.level_correct = 0
        self.question_index_level = 0
        self.feedback_active = False
        self.selected_answer = None
        self._refresh_selection_buttons()
        self.selected_label.config(text='Selected answer: None')
        self.feedback_label.config(text='')
        self.performance_label.config(text=self._performance_text())
        self._update_question_progress(QUESTION_DURATION_SEC)

        if level > 5:
            self.finish_test()
            return

        self.status_label.config(text=f'Level {level} / 5')
        self.instruction_label.config(
            text='Select the answer (0-9) using the mouse and press Submit.'
        )
        self.level_end_perf = time.perf_counter() + LEVEL_DURATION_SEC
        self._log('LEVEL_START', note=f'level_{level}')
        self.present_next_question()
        self.level_tick()

    def present_next_question(self):
        if time.perf_counter() >= self.level_end_perf:
            self.end_level()
            return

        self.selected_answer = None
        self._refresh_selection_buttons()
        self.selected_label.config(text='Selected answer: None')
        self.feedback_label.config(text='')

        self.question_index_global += 1
        self.question_index_level += 1
        self.current_question = self.generate_question(self.current_level)
        self.question_label.config(text=self.current_question.display)
        self.current_question_start_perf = time.perf_counter()
        self.question_deadline_perf = self.current_question_start_perf + QUESTION_DURATION_SEC
        self.question_active = True
        self.feedback_active = False
        self._set_input_state(True)
        self._update_question_progress(QUESTION_DURATION_SEC)
        self._log('QUESTION_ONSET')
        self.question_tick()

    def level_tick(self):
        if not self.experiment_started:
            return
        now = time.perf_counter()
        if self.level_end_perf is None:
            return

        level_left = max(0, self.level_end_perf - now)
        q_left = 0.0
        if self.question_deadline_perf is not None and self.question_active:
            q_left = max(0, self.question_deadline_perf - now)

        self.timer_label.config(text=f'Level time left: {level_left:0.1f}s')
        self._update_question_progress(q_left)

        if level_left <= 0:
            self.end_level()
            return

        self.level_tick_job = self.root.after(100, self.level_tick)

    def question_tick(self):
        if not self.question_active:
            return
        if time.perf_counter() >= self.question_deadline_perf:
            self.handle_timeout()
            return
        self.question_tick_job = self.root.after(100, self.question_tick)

    def select_answer(self, value):
        if not self.question_active or self.feedback_active:
            return
        self.selected_answer = value
        self.selected_label.config(text=f'Selected answer: {value}')
        self._refresh_selection_buttons()

    def _refresh_selection_buttons(self):
        for i, btn in enumerate(self.digit_buttons):
            btn.config(relief=tk.SUNKEN if self.selected_answer == i else tk.RAISED)

    def submit_answer(self):
        if not self.question_active or self.feedback_active:
            return

        if self.selected_answer is None:
            messagebox.showinfo('No answer selected', 'Please select a digit from 0 to 9 first.')
            return

        rt_ms = int((time.perf_counter() - self.current_question_start_perf) * 1000)
        is_correct = int(self.selected_answer == self.current_question.answer)

        self.question_active = False
        self.feedback_active = True
        self.total_answered += 1
        self.level_answered += 1

        if is_correct:
            self.total_correct += 1
            self.level_correct += 1

        self._log(
            'RESPONSE',
            selected_answer=self.selected_answer,
            is_correct=is_correct,
            rt_ms=rt_ms
        )
        self.show_feedback(is_correct, timeout=False)

    def handle_timeout(self):
        if not self.question_active:
            return

        self.question_active = False
        self.feedback_active = True
        self.total_answered += 1
        self.level_answered += 1
        self.selected_answer = None
        self.selected_label.config(text='Selected answer: None')
        self._refresh_selection_buttons()

        self._log(
            'TIMEOUT',
            selected_answer='',
            is_correct=0,
            rt_ms=QUESTION_DURATION_SEC * 1000
        )
        self.show_feedback(0, timeout=True)

    def show_feedback(self, is_correct, timeout=False):
        self._set_input_state(False)
        self._update_question_progress(0)

        if timeout:
            correctness_text = f'Time out. Correct answer: {self.current_question.answer}'
            color = 'red'
        elif is_correct:
            correctness_text = f'Correct. Answer: {self.current_question.answer}'
            color = 'green'
        else:
            correctness_text = f'Incorrect. Correct answer: {self.current_question.answer}'
            color = 'red'

        performance_text = self._performance_status_line()
        self.feedback_label.config(
            text=f'{correctness_text}\n{performance_text}',
            fg=color
        )
        self.performance_label.config(text=self._performance_text())
        self.next_question_job = self.root.after(FEEDBACK_MS, self.after_feedback)

    def after_feedback(self):
        self.feedback_active = False
        if time.perf_counter() >= self.level_end_perf:
            self.end_level()
        else:
            self.present_next_question()

    def end_level(self):
        self._cancel_question_only()
        self.question_active = False
        self.feedback_active = False
        self._set_input_state(False)
        self._update_question_progress(0)
        self._log(
            'LEVEL_END',
            note=f'level_{self.current_level}_accuracy_{self._current_accuracy():.2f}'
        )

        if self.current_level < 5:
            self.feedback_label.config(
                text=(
                    f'Level {self.current_level} finished.\n'
                    f'Level accuracy: {self._level_accuracy():.1f}%\n'
                    'Starting next level...'
                ),
                fg='black'
            )
            next_level = self.current_level + 1
            self.next_question_job = self.root.after(1500, lambda: self.start_level(next_level))
        else:
            self.finish_test()

    def finish_test(self):
        self._cancel_pending()
        self.experiment_started = False
        overall_accuracy = self._current_accuracy()
        status = 'ABOVE average' if overall_accuracy >= AVERAGE_PERFORMANCE else 'BELOW average'

        self.status_label.config(text='Test Completed')
        self.timer_label.config(text='')
        self.instruction_label.config(text='The experiment has ended.')
        self.question_label.config(text='')
        self.selected_label.config(text='Selected answer: None')
        self._set_input_state(False)
        self._update_question_progress(0)

        self.feedback_label.config(
            text=(
                f'Final accuracy: {overall_accuracy:.1f}%\n'
                f'Performance: {status}\n'
                f'Log saved to: {self.log_path.name}'
            ),
            fg='black'
        )
        self.performance_label.config(text=self._performance_text())
        self._log('EXPERIMENT_END', note=f'final_accuracy_{overall_accuracy:.2f}')

    def _current_accuracy(self):
        return 100.0 * self.total_correct / self.total_answered if self.total_answered else 0.0

    def _level_accuracy(self):
        return 100.0 * self.level_correct / self.level_answered if self.level_answered else 0.0

    def _performance_status_line(self):
        current = self._current_accuracy()
        if current >= AVERAGE_PERFORMANCE:
            return f'Performance above average ({current:.1f}% vs {AVERAGE_PERFORMANCE:.0f}%)'
        return f'Performance below average ({current:.1f}% vs {AVERAGE_PERFORMANCE:.0f}%)'

    def _performance_text(self):
        return (
            f'Overall accuracy: {self._current_accuracy():.1f}%    '
            f'Average target: {AVERAGE_PERFORMANCE:.0f}%'
        )

    def _set_input_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in self.digit_buttons:
            btn.config(state=state)
        self.submit_btn.config(state=state)

    def _cancel_question_only(self):
        if self.question_tick_job is not None:
            self.root.after_cancel(self.question_tick_job)
            self.question_tick_job = None
        if self.next_question_job is not None:
            self.root.after_cancel(self.next_question_job)
            self.next_question_job = None
        self.question_deadline_perf = None

    def _cancel_pending(self):
        self._cancel_question_only()
        if self.level_tick_job is not None:
            self.root.after_cancel(self.level_tick_job)
            self.level_tick_job = None

    def on_close(self):
        self._cancel_pending()
        try:
            self._log('WINDOW_CLOSED')
        except Exception:
            pass
        try:
            if self.log_file:
                self.log_file.flush()
                self.log_file.close()
        finally:
            self.root.destroy()

    def _to_display(self, expr: str) -> str:
        return expr.replace('//', '÷').replace('*', '×')

    def generate_question(self, level: int) -> Question:
        for _ in range(100000):
            if level == 1:
                # Category 1: only 2 one-digit numbers, only + or -
                a = random.randint(0, 9)
                b = random.randint(0, 9)
                op = random.choice(['+', '-'])

                value = a + b if op == '+' else a - b
                if 0 <= value <= 9:
                    expr_py = f'{a} {op} {b}'
                    expr_show = self._to_display(expr_py) + ' = ?'
                    return Question(expr_show, expr_py, value)

            elif level == 2:
                # Category 2: only 3 one-digit numbers, only + or -
                nums = [random.randint(0, 9) for _ in range(3)]
                ops = [random.choice(['+', '-']) for _ in range(2)]

                expr_py = f'{nums[0]} {ops[0]} {nums[1]} {ops[1]} {nums[2]}'
                try:
                    value = eval(expr_py)
                except Exception:
                    continue

                if isinstance(value, int) and 0 <= value <= 9:
                    expr_show = self._to_display(expr_py) + ' = ?'
                    return Question(expr_show, expr_py, value)

            elif level == 3:
                # Category 3: up to 4 integers, up to 2 can be 2-digit, only + or -
                n_terms = random.choice([3, 4])

                two_digit_count = random.choices(
                    population=[0, 1, 2],
                    weights=[1, 3, 2],
                    k=1
                )[0]
                two_digit_count = min(two_digit_count, n_terms)

                two_digit_positions = set(random.sample(range(n_terms), two_digit_count))
                nums = []
                for i in range(n_terms):
                    if i in two_digit_positions:
                        nums.append(random.randint(10, 99))
                    else:
                        nums.append(random.randint(0, 9))

                ops = [random.choice(['+', '-']) for _ in range(n_terms - 1)]

                expr_parts = [str(nums[0])]
                for op, n in zip(ops, nums[1:]):
                    expr_parts.extend([op, str(n)])

                expr_py = ' '.join(expr_parts)
                try:
                    value = eval(expr_py)
                except Exception:
                    continue

                if isinstance(value, int) and 0 <= value <= 9:
                    expr_show = self._to_display(expr_py) + ' = ?'
                    return Question(expr_show, expr_py, value)

            elif level == 4:
                # Category 4: up to 4 integers, up to 2 can be 2-digit, × allowed
                n_terms = random.choice([3, 4])

                two_digit_count = random.choices(
                    population=[0, 1, 2],
                    weights=[1, 3, 2],
                    k=1
                )[0]
                two_digit_count = min(two_digit_count, n_terms)

                two_digit_positions = set(random.sample(range(n_terms), two_digit_count))

                mult_pos = random.randint(0, n_terms - 2)
                mult_num_positions = {mult_pos, mult_pos + 1}

                nums = []
                for i in range(n_terms):
                    if i in two_digit_positions:
                        low, high = 10, 99
                    else:
                        low, high = 1, 9 if i in mult_num_positions else (0, 9)

                    if isinstance(low, tuple):
                        # safety fallback, not used
                        low, high = low

                    if i in mult_num_positions:
                        if i in two_digit_positions:
                            nums.append(random.randint(10, 99))
                        else:
                            nums.append(random.randint(1, 9))
                    else:
                        if i in two_digit_positions:
                            nums.append(random.randint(10, 99))
                        else:
                            nums.append(random.randint(0, 9))

                ops = []
                for i in range(n_terms - 1):
                    if i == mult_pos:
                        ops.append('*')
                    else:
                        ops.append(random.choice(['+', '-']))

                expr_parts = [str(nums[0])]
                for op, n in zip(ops, nums[1:]):
                    expr_parts.extend([op, str(n)])

                expr_py = ' '.join(expr_parts)
                try:
                    value = eval(expr_py)
                except Exception:
                    continue

                if isinstance(value, int) and 0 <= value <= 9:
                    expr_show = self._to_display(expr_py) + ' = ?'
                    return Question(expr_show, expr_py, value)

            elif level == 5:
                # Category 5: 4 integers, two-digit numbers allowed, both × and ÷ are used
                ans = random.randint(0, 9)
                op3 = random.choice(['+', '-'])

                if op3 == '+':
                    if ans == 0:
                        continue
                    d = random.randint(0, ans - 1)
                    q = ans - d
                else:
                    d = random.randint(0, 20)
                    q = ans + d

                if q <= 0:
                    continue

                c = random.randint(2, 30)
                total = q * c

                divisors = [x for x in range(1, min(99, total) + 1) if total % x == 0]
                if not divisors:
                    continue

                a = random.choice(divisors)
                b = total // a

                if not (1 <= a <= 99 and 1 <= b <= 99 and 1 <= c <= 99):
                    continue

                expr_py = f'{a} * {b} // {c} {op3} {d}'
                try:
                    value = eval(expr_py)
                except Exception:
                    continue

                if isinstance(value, int) and value == ans and 0 <= value <= 9:
                    expr_show = self._to_display(expr_py) + ' = ?'
                    return Question(expr_show, expr_py, value)

        raise RuntimeError(f'Could not generate a valid Level {level} question.')


def main():
    root = tk.Tk()
    app = MistApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
