"""Switch or Stay: a lesson on nonlinear neural networks.

Play the game, save every human decision to data/gameplay.csv, train the
networks on those rows, then compare a linear network with a ReLU network.
The comparison models live in models.py. The network you write, and the loop
that trains it, live in student_net.py.
"""

from __future__ import annotations

import os
import time
import uuid

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from data_logger import (
    DEFAULT_DATA_PATH,
    GameplayLogger,
    combination_counts,
    dataset_from_records,
    missing_successful_cases,
    reaction_summary,
)
from game import (
    CASES,
    RULE_TEXT,
    SEPARATION_TEXT,
    ScenarioDeck,
    action_name,
    correct_action,
    lane_name,
    resolve_round,
)
from models import (
    ARCHITECTURES,
    LINEAR_STACK_EXPLANATION,
    PRIMARY_LINEAR,
    PRIMARY_NONLINEAR,
    architecture_by_key,
    count_parameters,
)
from student_loader import (
    STUDENT_NET_PATH,
    check_trained_model,
    describe_student_error,
    load_student_train,
    loss_as_float,
)
from training import (
    EPOCHS,
    LEARNING_RATE,
    SEED,
    TrainHistory,
    conflicting_inputs,
    missing_combinations,
    predict_cases,
    set_seed,
    switch_probability,
    train_architecture,
)
from visualisation import LossSeries, render_decision_figure, render_loss_figure


NAV_H = 56
STATUS_H = 30
FOOTER_H = 34

BG = (18, 20, 26)
PANEL = (28, 32, 42)
PANEL_2 = (36, 41, 54)
LINE = (58, 66, 82)
TEXT = (236, 238, 244)
MUTED = (154, 162, 178)
ACCENT = (124, 108, 255)
BLUE = (76, 141, 255)
ORANGE = (255, 138, 61)
GREEN = (64, 201, 140)
RED = (255, 96, 104)
AMBER = (245, 197, 66)
WHITE = (244, 246, 250)
ASPHALT = (34, 38, 46)
SHOULDER = (22, 44, 38)

STUDENT_MODE = "student"
STUDENT_EPOCH_FPS = 50

AI_LOOK_SECONDS = 0.45
ANIM_SECONDS = 0.16
FLASH_SECONDS = 0.28


class TrainingCancelled(BaseException):
    """Raised when Esc is pressed during training. Not an Exception, so a
    student `except Exception` does not swallow the quit."""


class HitBox:
    def __init__(self, rect, action):
        self.rect = pygame.Rect(rect)
        self.action = action


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def smoothstep(amount: float) -> float:
    amount = max(0.0, min(1.0, amount))
    return amount * amount * (3.0 - 2.0 * amount)


def wrap_text(text: str, font: pygame.font.Font, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.size(trial)[0] <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class App:
    def __init__(self, data_path=None, size=None):
        pygame.init()
        if size is None:
            info = pygame.display.Info()
            width = min(1360, max(1024, info.current_w - 40))
            height = min(840, max(640, info.current_h - 80))
            size = (width, height)
        self.width, self.height = size
        pygame.display.set_caption("Switch or Stay")
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("segoeui", 24, bold=True)
        self.font_heading = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_body = pygame.font.SysFont("segoeui", 18)
        self.font_button = pygame.font.SysFont("segoeui", 16, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 16)
        self.font_tiny = pygame.font.SysFont("segoeui", 14)
        self.font_big = pygame.font.SysFont("segoeui", 40, bold=True)

        self.logger = GameplayLogger(data_path or DEFAULT_DATA_PATH)
        self.deck = ScenarioDeck()
        self.session_id = uuid.uuid4().hex
        self.screen_name = "play"
        self.mode = "human"
        self.include_mistakes = False
        self.models = {}
        self.histories = {}
        self.parameter_counts = {
            architecture.key: count_parameters(architecture.factory())
            for architecture in ARCHITECTURES
        }
        self.ai_stats = {architecture.key: {"rounds": 0, "correct": 0} for architecture in ARCHITECTURES}
        self.ai_stats[STUDENT_MODE] = {"rounds": 0, "correct": 0}
        self.student_losses: list[float] = []
        self.student_epoch = 0
        self.student_epoch_total = EPOCHS
        self.student_error = ""
        self.human_rounds = 0
        self.human_correct = 0
        self.status = "Human play. Press A to stay, S to switch. Your actions are saved."
        self.busy = False
        self.progress = 0.0
        self.running = True
        self.hitboxes: list[HitBox] = []
        self.plot_cache = {}
        self.plot_version = 0
        self.dataset_note = "No training run yet."
        self.figures_note = ""
        self.compare_keys = [item.key for item in ARCHITECTURES if item.key != PRIMARY_LINEAR]
        self.compare_index = self.compare_keys.index(PRIMARY_NONLINEAR)
        self._reaction_paused_at = None
        self.model_keys = {
            getattr(pygame, f"K_{index}"): architecture.key
            for index, architecture in enumerate(ARCHITECTURES, start=1)
        }
        self._begin_round()

    # --- round flow -------------------------------------------------------

    def _begin_round(self):
        self.car_lane, self.obstacle_lane = self.deck.next_scenario()
        self.phase = "decide"
        self.phase_t = 0.0
        self.shown_at = time.perf_counter()
        self._reaction_paused_at = None
        self.action = None
        self.committed = False
        self.result_success = None
        self.final_lane = self.car_lane
        self.car_draw = float(self.car_lane)
        self.anim_from = float(self.car_lane)
        self.anim_to = float(self.car_lane)
        self.obstacle_t = 0.0
        self.live_probability = None
        self.live_action = None

    def _commit(self, action: int, probability: float | None):
        if self.committed or self.phase != "decide":
            return
        self.committed = True
        resolved = resolve_round(self.car_lane, self.obstacle_lane, action, self.mode)
        self.action = resolved.player_action
        self.result_success = resolved.success
        self.final_lane = resolved.final_lane
        self.live_action = resolved.player_action
        self.live_probability = probability
        self.anim_from = float(self.car_lane)
        self.anim_to = float(resolved.final_lane)
        if resolved.record_for_training:
            reaction_ms = max(0, int((time.perf_counter() - self.shown_at) * 1000))
            try:
                self.logger.log_human_demonstration(
                    session_id=self.session_id,
                    car_lane=resolved.car_lane,
                    obstacle_lane=resolved.obstacle_lane,
                    player_action=resolved.player_action,
                    correct_action=resolved.correct_action,
                    success=resolved.success,
                    reaction_time_ms=reaction_ms,
                )
            except OSError as error:
                self.status = f"Could not save the round. Close the CSV if it is open elsewhere. {error}"
            else:
                self.human_rounds += 1
                self.human_correct += int(resolved.success)
                if resolved.success:
                    self.status = "Saved. Safe. This demonstration can train the networks."
                else:
                    self.status = "Saved the miss. It stays in the CSV. Default training skips misses."
        else:
            stats = self.ai_stats[self.mode]
            stats["rounds"] += 1
            stats["correct"] += int(resolved.success)
            name = self._mode_name(self.mode)
            verdict = "right" if resolved.success else "wrong"
            self.status = (
                f"{name} chose {action_name(action)} and was {verdict}. "
                "AI rounds are not added to the training file."
            )
        self.phase = "animate"
        self.phase_t = 0.0

    def _human_action(self, action: int):
        if self.screen_name != "play" or self.phase != "decide":
            return
        if self.mode != "human":
            self.status = "The network is driving. Press H to take the car back."
            return
        self._commit(action, None)

    def _mode_name(self, mode: str) -> str:
        if mode == "human":
            return "Human"
        if mode == STUDENT_MODE:
            return "Your network"
        return architecture_by_key(mode).name

    def set_mode(self, mode: str):
        if mode != "human" and mode not in self.models:
            if mode == STUDENT_MODE:
                self.status = "Train the network in student_net.py first. Press G on the Your net screen."
                self._set_screen("student")
            else:
                name = architecture_by_key(mode).name
                self.status = f"Train the models first, then let {name} drive."
                self._set_screen("train")
            return
        self.mode = mode
        if self.phase == "decide":
            self.phase_t = 0.0
            self.shown_at = time.perf_counter()
            self._reaction_paused_at = None
        if mode == "human":
            self.status = "Human play. Your actions are recorded."
        else:
            self.status = f"{self._mode_name(mode)} is driving. These rounds are not saved."
        self._set_screen("play")

    def _set_screen(self, name: str):
        if name == self.screen_name:
            return
        if self.screen_name == "play" and self.phase == "decide" and self.mode == "human":
            self._reaction_paused_at = time.perf_counter()
        if name == "play" and self._reaction_paused_at is not None:
            self.shown_at += time.perf_counter() - self._reaction_paused_at
            self._reaction_paused_at = None
        self.screen_name = name

    def toggle_mistakes(self):
        self.include_mistakes = not self.include_mistakes
        if self.include_mistakes:
            self.status = "Next training run imitates every recorded action, including misses."
        else:
            self.status = "Next training run uses successful demonstrations only."

    def clear_ai_stats(self, message: str | None = None):
        for key in self.ai_stats:
            self.ai_stats[key] = {"rounds": 0, "correct": 0}
        if message:
            self.status = message

    def reset_ai_stats(self):
        self.clear_ai_stats("AI scores cleared. Human demonstrations in the CSV were kept.")

    def cycle_compare(self, step: int):
        self.compare_index = (self.compare_index + step) % len(self.compare_keys)

    def train(self):
        if self.busy:
            return
        features, labels = dataset_from_records(
            self.logger.records,
            successes_only=not self.include_mistakes,
        )
        if len(labels) == 0:
            self._set_screen("train")
            if self.logger.records and not self.include_mistakes:
                self.status = "No successful demonstrations yet. Play a correct round, or include mistakes."
            else:
                self.status = "No demonstrations yet. Play the game in human mode first."
            return

        self.busy = True
        self.progress = 0.0
        kept_model = self.models.get(STUDENT_MODE)
        kept_history = self.histories.get(STUDENT_MODE)
        self.models = {}
        self.histories = {}
        if kept_model is not None and kept_history is not None:
            self.models[STUDENT_MODE] = kept_model
            self.histories[STUDENT_MODE] = kept_history
        self.plot_cache.clear()
        self.figures_note = ""
        self._set_screen("train")
        filter_text = (
            "successful demonstrations only"
            if not self.include_mistakes
            else "every recorded action, including misses"
        )
        self.dataset_note = f"{len(labels)} rows, {filter_text}."
        self.status = "Training..."
        self._paint()
        cancelled = False
        try:
            model_count = len(ARCHITECTURES)
            for index, architecture in enumerate(ARCHITECTURES):

                def on_epoch(epoch, total, loss, accuracy, index=index, architecture=architecture):
                    self.progress = (index + (epoch + 1) / total) / model_count
                    self.status = (
                        f"Training {architecture.name}: epoch {epoch + 1} of {total}"
                        f"    loss {loss:.3f}    accuracy {accuracy:.0%}"
                    )
                    self._paint()
                    if self._quit_requested():
                        raise TrainingCancelled()

                network, history = train_architecture(architecture, features, labels, on_epoch=on_epoch)
                self.models[architecture.key] = network
                self.histories[architecture.key] = history
                self._paint()
        except TrainingCancelled:
            cancelled = True
        except Exception as error:
            self.status = f"Training stopped: {error}"
            cancelled = True
        finally:
            self.busy = False
            self.progress = 0.0
            if self.mode != "human" and self.mode not in self.models:
                self.mode = "human"
        if cancelled or not self.running:
            if self.running and not self.status.startswith("Training stopped"):
                self.status = "Training cancelled."
            return

        self.plot_version += 1
        for architecture in ARCHITECTURES:
            self.ai_stats[architecture.key] = {"rounds": 0, "correct": 0}
        try:
            self._save_figures()
        except OSError as error:
            self.figures_note = f"Could not write the PNG files ({error})."
        else:
            folder = self.logger.path.parent
            self.figures_note = "Plots saved in the data folder: loss_curves.png and decision_boundaries.png."
        linear = self.histories[PRIMARY_LINEAR]
        relu = self.histories[PRIMARY_NONLINEAR]
        if linear.uncommitted():
            linear_result = "Linear stayed near 50% on every situation."
        else:
            linear_result = f"Linear solved {linear.cases_correct}/4 situations."
        self.status = (
            f"Trained on {linear.n_samples} rows. "
            f"{linear_result} "
            f"ReLU solved {relu.cases_correct}/4."
        )
        if linear.missing_combinations:
            missing = ", ".join(
                f"car {lane_name(car)} / obstacle {lane_name(obstacle)}"
                for car, obstacle in linear.missing_combinations
            )
            self.status += f" Missing from training: {missing}."

    def train_student(self):
        """Read student_net.py again and run the train() written in that file."""
        if self.busy:
            return
        features, labels = dataset_from_records(
            self.logger.records,
            successes_only=not self.include_mistakes,
        )
        self._set_screen("student")
        if len(labels) == 0:
            if self.logger.records and not self.include_mistakes:
                self.status = "No successful demonstrations yet. Play a correct round, or include mistakes."
            else:
                self.status = "Play the game first. Your network trains on the keys you pressed."
            return

        self.busy = True
        self.progress = 0.0
        self.student_losses = []
        self.student_epoch = 0
        self.student_epoch_total = 0
        self.student_error = ""
        self.status = f"Reading {STUDENT_NET_PATH.name}..."
        self._paint()
        try:
            train = load_student_train(STUDENT_NET_PATH)
            set_seed(SEED)

            def report(loss):
                value = loss_as_float(loss)
                self.student_losses.append(value)
                self.student_epoch = len(self.student_losses)
                self.student_epoch_total = self.student_epoch
                self.status = f"Your network    epoch {self.student_epoch}    loss {value:.3f}"
                self._paint()
                self.clock.tick(STUDENT_EPOCH_FPS)
                if self._quit_requested():
                    raise TrainingCancelled()

            model = check_trained_model(train(features, labels, report))
            history = TrainHistory(
                losses=list(self.student_losses),
                training_loss=self.student_losses[-1] if self.student_losses else 0.0,
                n_samples=int(len(labels)),
                missing_combinations=missing_combinations(features),
                conflicting_inputs=conflicting_inputs(features, labels),
                predictions=predict_cases(model),
            )
        except TrainingCancelled:
            self.status = "Training cancelled."
            return
        except Exception as error:
            self.student_error = describe_student_error(error)
            self.status = "Fix student_net.py, save the file, and press G again."
            return
        finally:
            self.busy = False
            self.progress = 0.0

        self.models[STUDENT_MODE] = model
        self.histories[STUDENT_MODE] = history
        self.ai_stats[STUDENT_MODE] = {"rounds": 0, "correct": 0}
        if history.uncommitted():
            result = "It stayed near 50%. A straight network often does that on this game."
        else:
            result = f"It matched the rule on {history.cases_correct} of 4 situations."
        reported = ""
        if not self.student_losses:
            reported = " Call report(loss) once per epoch to draw the line."
        self.status = f"Loss ended at {history.training_loss:.3f}. {result} Press U to let it drive.{reported}"

    def _save_figures(self):
        folder = self.logger.path.parent
        loss_image = render_loss_figure(self._loss_series(), folder / "loss_curves.png")
        right_key = PRIMARY_NONLINEAR if PRIMARY_NONLINEAR in self.models else self.compare_keys[0]
        decision_image = render_decision_figure(
            self.models[PRIMARY_LINEAR],
            self._model_title(PRIMARY_LINEAR),
            self.models[right_key],
            self._model_title(right_key),
            folder / "decision_boundaries.png",
        )
        self.plot_cache[("loss", self.plot_version)] = self._rgb_surface(loss_image)
        self.plot_cache[("decision", self.plot_version, right_key)] = self._rgb_surface(decision_image)

    def _loss_series(self) -> list[LossSeries]:
        series = []
        for architecture in ARCHITECTURES:
            history = self.histories.get(architecture.key)
            if history is None:
                continue
            series.append(
                LossSeries(
                    name=architecture.name,
                    color=architecture.color,
                    losses=history.losses,
                    accuracies=history.accuracies,
                    dashed=architecture.linear_family,
                )
            )
        return series

    def _model_title(self, key: str) -> str:
        architecture = architecture_by_key(key)
        history = self.histories.get(key)
        if history is None:
            return architecture.name
        return (
            f"{architecture.name}   {history.cases_correct}/4 situations"
            f"   loss {history.training_loss:.3f}"
        )

    # --- events -----------------------------------------------------------

    def _quit_requested(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return True
        return False

    def _handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.busy:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for hitbox in self.hitboxes:
                if hitbox.rect.collidepoint(event.pos):
                    hitbox.action()
                    return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_t:
            self.train()
        elif key == pygame.K_y:
            self._set_screen("student")
        elif key == pygame.K_g:
            self.train_student()
        elif key == pygame.K_u:
            self.set_mode(STUDENT_MODE)
        elif key == pygame.K_p:
            self._set_screen("play")
        elif key == pygame.K_d:
            self._set_screen("data")
        elif key == pygame.K_c:
            self._set_screen("curves")
        elif key == pygame.K_v:
            self._set_screen("boundary")
        elif key == pygame.K_i and self.screen_name in {"train", "curves"}:
            self.toggle_mistakes()
        elif key == pygame.K_r and self.screen_name == "play":
            self.reset_ai_stats()
        elif self.screen_name == "boundary" and key in {pygame.K_LEFT, pygame.K_RIGHT}:
            self.cycle_compare(-1 if key == pygame.K_LEFT else 1)
        elif self.screen_name == "play":
            self._handle_play_key(key)

    def _handle_play_key(self, key):
        if key in self.model_keys:
            self.set_mode(self.model_keys[key])
        elif key == pygame.K_h:
            self.set_mode("human")
        elif key == pygame.K_l:
            self.set_mode(PRIMARY_LINEAR)
        elif key == pygame.K_n:
            self.set_mode(PRIMARY_NONLINEAR)
        elif key in {pygame.K_a, pygame.K_s}:
            self._human_action(0 if key == pygame.K_a else 1)

    def update(self, dt: float):
        if self.busy or self.screen_name != "play":
            return
        if self.phase == "decide":
            if self.mode == "human" or self.mode not in self.models:
                return
            self.live_probability = switch_probability(
                self.models[self.mode], self.car_lane, self.obstacle_lane
            )
            self.live_action = 1 if self.live_probability >= 0.5 else 0
            self.phase_t += dt
            if self.phase_t >= AI_LOOK_SECONDS:
                self._commit(self.live_action, self.live_probability)
            return
        if self.phase == "animate":
            self.phase_t += dt
            amount = smoothstep(self.phase_t / ANIM_SECONDS)
            self.car_draw = self.anim_from + (self.anim_to - self.anim_from) * amount
            self.obstacle_t = amount
            if self.phase_t >= ANIM_SECONDS:
                self.car_draw = self.anim_to
                self.obstacle_t = 1.0
                self.phase = "flash"
                self.phase_t = 0.0
            return
        if self.phase == "flash" and self.phase_t + dt >= FLASH_SECONDS:
            self._begin_round()
        else:
            self.phase_t += dt

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self._handle_event(event)
            self.update(dt)
            self._paint()
        pygame.quit()

    # --- drawing helpers --------------------------------------------------

    def _paint(self):
        self.hitboxes = []
        self.draw()
        pygame.display.flip()

    def content_rect(self) -> pygame.Rect:
        top = NAV_H + STATUS_H + 8
        bottom = self.height - FOOTER_H - 8
        return pygame.Rect(16, top, self.width - 32, bottom - top)

    def blit(self, text, pos, font, color=TEXT):
        self.screen.blit(font.render(text, True, color), pos)

    def blit_fit(self, text, pos, font, color, max_width):
        if font.size(text)[0] <= max_width:
            self.blit(text, pos, font, color)
            return
        trimmed = text
        while trimmed and font.size(trimmed + "...")[0] > max_width:
            trimmed = trimmed[:-1]
        self.blit(trimmed + "...", pos, font, color)

    def blit_center(self, text, rect, font, color):
        image = font.render(text, True, color)
        x = rect.x + (rect.w - image.get_width()) // 2
        y = rect.y + (rect.h - image.get_height()) // 2
        self.screen.blit(image, (x, y))

    def draw_wrapped(self, text, x, y, font, color, max_width, gap=2) -> int:
        for line in wrap_text(text, font, max_width):
            image = font.render(line, True, color)
            self.screen.blit(image, (x, y))
            y += image.get_height() + gap
        return y

    def panel(self, rect, fill=PANEL):
        pygame.draw.rect(self.screen, fill, rect, border_radius=14)
        pygame.draw.rect(self.screen, LINE, rect, width=1, border_radius=14)

    def button(self, rect, label, action, fill, text_color=TEXT, enabled=True, border=None):
        rect = pygame.Rect(rect)
        if rect.w <= 0 or rect.h <= 0:
            return rect
        hover = bool(enabled and action is not None and rect.collidepoint(pygame.mouse.get_pos()))
        if not enabled:
            fill = tuple(max(32, channel // 2) for channel in fill)
        colour = tuple(min(255, channel + 24) for channel in fill) if hover else fill
        pygame.draw.rect(self.screen, colour, rect, border_radius=8)
        if border is not None:
            pygame.draw.rect(self.screen, border, rect, width=2, border_radius=8)
        font = self.font_tiny if rect.h < 32 else self.font_button
        self.blit_center(label, rect, font, text_color if enabled else MUTED)
        if enabled and action is not None:
            self.hitboxes.append(HitBox(rect, action))
        return rect

    def _rgb_surface(self, image) -> pygame.Surface:
        height, width = image.shape[:2]
        raw = image.tobytes()
        return pygame.image.frombuffer(raw, (width, height), "RGB").convert()

    def _fit_surface(self, surface, rect) -> pygame.Surface:
        scale = min(rect.w / surface.get_width(), rect.h / surface.get_height())
        size = (max(1, int(surface.get_width() * scale)), max(1, int(surface.get_height() * scale)))
        return pygame.transform.smoothscale(surface, size)

    def draw(self):
        self.screen.fill(BG)
        self.draw_nav()
        self.draw_status()
        if self.screen_name == "play":
            self.draw_play()
        elif self.screen_name == "data":
            self.draw_data()
        elif self.screen_name == "train":
            self.draw_train()
        elif self.screen_name == "curves":
            self.draw_curves()
        elif self.screen_name == "student":
            self.draw_student()
        else:
            self.draw_boundary()
        self.draw_footer()
        hovering = any(hitbox.rect.collidepoint(pygame.mouse.get_pos()) for hitbox in self.hitboxes)
        cursor = pygame.SYSTEM_CURSOR_HAND if hovering else pygame.SYSTEM_CURSOR_ARROW
        pygame.mouse.set_cursor(cursor)

    def draw_nav(self):
        pygame.draw.rect(self.screen, (14, 16, 22), (0, 0, self.width, NAV_H))
        self.blit("Switch or Stay", (18, 4), self.font_title)
        self.blit("Play, collect, train, compare", (18, 32), self.font_tiny, MUTED)
        items = [
            ("play", "Play", 84),
            ("data", "Data", 84),
            ("train", "Train", 84),
            ("curves", "Loss curves", 128),
            ("boundary", "Boundaries", 132),
            ("student", "Your net", 110),
        ]
        total = sum(width for _key, _label, width in items) + 8 * (len(items) - 1)
        x = self.width - 16 - total
        for key, label, width in items:
            fill = ACCENT if self.screen_name == key else PANEL
            self.button((x, 10, width, 36), label, lambda name=key: self._set_screen(name), fill=fill)
            x += width + 8

    def draw_status(self):
        y = NAV_H
        pygame.draw.rect(self.screen, (24, 27, 36), (0, y, self.width, STATUS_H))
        colour = AMBER if self.busy else TEXT
        self.blit_fit(self.status, (18, y + 6), self.font_small, colour, self.width - 36)
        if self.busy:
            pygame.draw.rect(self.screen, ACCENT, (0, y + STATUS_H - 3, int(self.width * self.progress), 3))

    def draw_footer(self):
        y = self.height - FOOTER_H
        pygame.draw.rect(self.screen, (14, 16, 22), (0, y, self.width, FOOTER_H))
        hints = {
            "play": "A stay   S switch   Y your net   G train it   U drive it   L linear   N ReLU   T compare all",
            "student": "Edit student_net.py and save.   G train, watch the loss   U drive the car   Esc quits",
            "data": "A success in every card is enough to train.   P play   T train   Esc quits",
            "train": "T train   I successes only / include mistakes   C loss curves   V boundaries   Esc quits",
            "curves": "Dashed curves have no hidden activation.   V boundaries   T train again   Esc quits",
            "boundary": "Left and right arrows change the right-hand network.   C loss curves   Esc quits",
        }
        self.blit_fit(hints[self.screen_name], (18, y + 8), self.font_tiny, MUTED, self.width - 36)

    # --- play -------------------------------------------------------------

    def draw_play(self):
        area = self.content_rect()
        left_w = int(area.w * 0.48)
        left = pygame.Rect(area.x, area.y, left_w, area.h)
        right = pygame.Rect(left.right + 12, area.y, area.w - left_w - 12, area.h)
        self.draw_road(left)
        self.draw_hud(right)

    def draw_road(self, panel_rect: pygame.Rect):
        pygame.draw.rect(self.screen, SHOULDER, panel_rect, border_radius=14)
        pygame.draw.rect(self.screen, LINE, panel_rect, width=1, border_radius=14)
        if self.phase == "decide":
            headline, colour = "A stay     S switch", TEXT
        elif self.phase == "animate":
            headline = "Switching" if self.action == 1 else "Staying"
            colour = ORANGE if self.action == 1 else BLUE
        else:
            headline = "Safe" if self.result_success else "Crash"
            colour = GREEN if self.result_success else RED
        self.blit_fit(headline, (panel_rect.x + 16, panel_rect.y + 10), self.font_heading, colour, panel_rect.w - 32)
        asphalt = pygame.Rect(panel_rect.x + 18, panel_rect.y + 46, panel_rect.w - 36, panel_rect.h - 64)
        pygame.draw.rect(self.screen, ASPHALT, asphalt, border_radius=8)
        lane_w = asphalt.w / 2
        tint = pygame.Surface((int(lane_w), asphalt.h), pygame.SRCALPHA)
        tint.fill((255, 138, 61, 36))
        self.screen.blit(tint, (asphalt.x + int(lane_w) * self.obstacle_lane, asphalt.y))
        pygame.draw.line(self.screen, WHITE, (asphalt.x + 8, asphalt.y + 8), (asphalt.x + 8, asphalt.bottom - 8), 3)
        pygame.draw.line(
            self.screen, WHITE, (asphalt.right - 8, asphalt.y + 8), (asphalt.right - 8, asphalt.bottom - 8), 3
        )
        dash_x = int(asphalt.centerx)
        y = asphalt.y + 10
        while y < asphalt.bottom - 16:
            pygame.draw.rect(self.screen, AMBER, (dash_x - 3, y, 6, 16), border_radius=2)
            y += 30
        for lane, label in ((0, "LEFT"), (1, "RIGHT")):
            center = asphalt.x + lane_w * lane + lane_w / 2
            image = self.font_tiny.render(label, True, MUTED)
            self.screen.blit(image, image.get_rect(center=(center, asphalt.y + 16)))

        car_y = asphalt.bottom - 62
        obstacle_y = asphalt.y + 48 + (car_y - 8 - (asphalt.y + 48)) * self.obstacle_t
        obstacle_x = asphalt.x + lane_w * self.obstacle_lane + lane_w / 2
        car_x = asphalt.x + lane_w * self.car_draw + lane_w / 2
        self.draw_obstacle(obstacle_x, obstacle_y)
        self.draw_car(car_x, car_y)
        if self.phase == "flash":
            word = "SAFE" if self.result_success else "CRASH"
            word_colour = GREEN if self.result_success else RED
            image = self.font_big.render(word, True, word_colour)
            self.screen.blit(image, image.get_rect(center=(asphalt.centerx, asphalt.y + 78)))

    def draw_car(self, center_x: float, center_y: float):
        body = pygame.Rect(int(center_x) - 22, int(center_y) - 38, 44, 76)
        pygame.draw.rect(self.screen, BLUE, body, border_radius=12)
        pygame.draw.rect(self.screen, WHITE, body, width=2, border_radius=12)
        roof = pygame.Rect(body.x + 7, body.y + 16, 30, 28)
        pygame.draw.rect(self.screen, (48, 104, 196), roof, border_radius=7)
        pygame.draw.rect(self.screen, (214, 230, 244), (body.x + 8, body.y + 6, 28, 14), border_radius=4)
        pygame.draw.circle(self.screen, (255, 236, 180), (body.x + 10, body.y + 8), 3)
        pygame.draw.circle(self.screen, (255, 236, 180), (body.right - 10, body.y + 8), 3)

    def draw_obstacle(self, center_x: float, center_y: float):
        rect = pygame.Rect(int(center_x) - 28, int(center_y) - 22, 56, 44)
        pygame.draw.rect(self.screen, (196, 64, 56), rect, border_radius=6)
        pygame.draw.rect(self.screen, (255, 196, 64), (rect.x, rect.centery - 7, rect.w, 14))
        pygame.draw.rect(self.screen, WHITE, rect, width=2, border_radius=6)

    def _score_row(self, x, y, width, row_h, key_label, name, mode, colour):
        stats = self.ai_stats.get(mode, {"rounds": 0, "correct": 0})
        if mode not in self.models:
            text = "not trained"
        elif stats["rounds"]:
            text = f"{stats['correct']}/{stats['rounds']}   {stats['correct'] / stats['rounds']:.0%}"
        else:
            text = "no rounds yet"
        if self.mode == mode:
            pygame.draw.rect(self.screen, PANEL_2, (x, y, width, row_h), border_radius=6)
        pygame.draw.rect(self.screen, colour, (x, y + 4, 8, row_h - 8), border_radius=2)
        self.blit_fit(f"{key_label}  {name}", (x + 14, y + 2), self.font_tiny, TEXT, int(width * 0.55))
        self.blit_fit(text, (x + int(width * 0.58), y + 2), self.font_tiny, MUTED, int(width * 0.4))

    def draw_hud(self, panel_rect: pygame.Rect):
        self.panel(panel_rect)
        x = panel_rect.x + 16
        y = panel_rect.y + 12
        width = panel_rect.w - 32
        if self.mode == "human":
            pygame.draw.circle(self.screen, RED, (x + 7, y + 8), 6)
            self.blit("Recording human demonstrations", (x + 20, y), self.font_small, RED)
        else:
            pygame.draw.circle(self.screen, MUTED, (x + 7, y + 8), 6)
            self.blit("AI play is not written to the CSV", (x + 20, y), self.font_small, MUTED)
        y += 28
        mode_name = self._mode_name(self.mode) if self.mode == "human" else self._mode_name(self.mode) + " AI"
        self.blit(mode_name, (x, y), self.font_heading)
        y += 28
        if self.mode == "human":
            correct, rounds = self.human_correct, self.human_rounds
            score_label = "Your session"
        else:
            stats = self.ai_stats[self.mode]
            correct, rounds = stats["correct"], stats["rounds"]
            score_label = "This AI"
        accuracy = f"{correct / rounds:.0%}" if rounds else "-"
        self.blit(f"{score_label}:  {correct} / {rounds} correct    {accuracy}", (x, y), self.font_body)
        y += 28
        self.blit(
            f"Car {lane_name(self.car_lane)}      Obstacle {lane_name(self.obstacle_lane)}",
            (x, y),
            self.font_body,
        )
        y += 24
        self.blit_fit(RULE_TEXT, (x, y), self.font_tiny, MUTED, width)
        y += 24
        self.draw_prediction_box(pygame.Rect(x, y, width, 92))
        y += 102
        self.blit("AI evaluation, kept separate from the CSV", (x, y), self.font_tiny, MUTED)
        y += 20
        row_h = 22
        self._score_row(x, y, width, row_h, "Y", "Your network", STUDENT_MODE, (255, 176, 92))
        y += row_h
        for index, architecture in enumerate(ARCHITECTURES, start=1):
            self._score_row(
                x, y, width, row_h, str(index), architecture.name, architecture.key, hex_rgb(architecture.color)
            )
            y += row_h
        y += 8
        missing = missing_successful_cases(self.logger.records)
        if not self.logger.records:
            coverage = "No demonstrations yet."
            coverage_colour = MUTED
        elif not missing:
            coverage = f"Ready to train. {len(self.logger.records)} saved rounds cover all four situations."
            coverage_colour = GREEN
        else:
            pending = "; ".join(f"car {lane_name(car)}, obstacle {lane_name(obs)}" for car, obs in missing)
            coverage = "Still need a success for " + pending
            coverage_colour = AMBER
        y = self.draw_wrapped(coverage, x, y, self.font_tiny, coverage_colour, width)
        y += 8
        button_w = (width - 10) // 2
        human = self.mode == "human" and self.phase == "decide"
        self.button((x, y, button_w, 40), "A   Stay", lambda: self._human_action(0), fill=BLUE, enabled=human)
        self.button(
            (x + button_w + 10, y, button_w, 40),
            "S   Switch",
            lambda: self._human_action(1),
            fill=ORANGE,
            text_color=(28, 20, 16),
            enabled=human,
        )

    def draw_prediction_box(self, rect: pygame.Rect):
        self.panel(rect, PANEL_2)
        if self.mode == "human" or self.live_probability is None:
            title = "You are driving" if self.mode == "human" else "Reading the network..."
            self.blit(title, (rect.x + 12, rect.y + 10), self.font_small)
            self.draw_wrapped(
                "The orange lane has the obstacle. Switch only if your car is in that lane.",
                rect.x + 12,
                rect.y + 36,
                self.font_tiny,
                MUTED,
                rect.w - 24,
            )
            return
        if abs(self.live_probability - 0.5) < 0.05:
            chance = "about 50%"
            action = "tie, counted as " + action_name(self.live_action if self.live_action is not None else 0)
        else:
            chance = f"{self.live_probability:.0%}"
            action = action_name(self.live_action if self.live_action is not None else 0).upper()
        self.blit_fit(
            f"P(switch) {chance}    {action}",
            (rect.x + 12, rect.y + 8),
            self.font_small,
            TEXT,
            rect.w - 24,
        )
        bar = pygame.Rect(rect.x + 12, rect.y + 36, rect.w - 24, 22)
        pygame.draw.rect(self.screen, (16, 18, 24), bar, border_radius=6)
        fill_w = int(bar.w * min(1.0, max(0.0, self.live_probability)))
        fill = ORANGE if self.live_probability >= 0.5 else BLUE
        if fill_w:
            previous_clip = self.screen.get_clip()
            self.screen.set_clip(bar)
            pygame.draw.rect(self.screen, fill, (bar.x, bar.y, max(fill_w, 8), bar.h), border_radius=6)
            self.screen.set_clip(previous_clip)
        mark = bar.x + bar.w // 2
        pygame.draw.line(self.screen, WHITE, (mark, bar.y - 3), (mark, bar.bottom + 3), 2)
        self.blit("50% threshold. At or above it, the car switches.", (rect.x + 12, rect.y + 64), self.font_tiny, MUTED)

    # --- student network --------------------------------------------------

    def draw_student(self):
        area = self.content_rect()
        self.blit("Your network", (area.x, area.y), self.font_heading)
        self.blit_fit(str(STUDENT_NET_PATH), (area.x + 180, area.y + 4), self.font_tiny, MUTED, area.w - 180)
        chart = pygame.Rect(area.x, area.y + 36, int(area.w * 0.62), area.h - 36)
        side = pygame.Rect(chart.right + 12, chart.y, area.right - chart.right - 12, chart.h)
        losses = self.student_losses
        history = self.histories.get(STUDENT_MODE)
        if not self.busy and history is not None:
            losses = history.losses
        self.draw_live_loss(chart, losses)
        self.blit("Write it, save, train", (side.x, side.y), self.font_small)
        y = side.y + 28
        y = self.draw_wrapped(
            "Open student_net.py. Write StudentNet, then write train(features, labels, report). "
            "Call report(loss) once each epoch and return the network. Save the file.",
            side.x,
            y,
            self.font_tiny,
            MUTED,
            side.w,
        )
        y += 8
        if self.student_error:
            y = self.draw_wrapped(self.student_error, side.x, y, self.font_tiny, AMBER, side.w)
            y += 6
        if self.busy or self.student_epoch:
            self.blit(
                f"Epoch {self.student_epoch}",
                (side.x, y),
                self.font_tiny,
                MUTED,
            )
            y += 22
        if losses:
            self.blit(f"Loss {losses[-1]:.3f}", (side.x, y), self.font_big, ORANGE)
            y += 52
        if history is not None and not self.busy:
            if history.uncommitted():
                summary = "Still about 50% on every corner. A network with no bend often ends here."
                colour = AMBER
            else:
                summary = f"Matched the rule on {history.cases_correct} of the 4 situations."
                colour = GREEN if history.cases_correct == 4 else TEXT
            y = self.draw_wrapped(summary, side.x, y, self.font_tiny, colour, side.w)
            y += 8
            for prediction in history.predictions:
                if abs(prediction.probability - 0.5) < 0.05:
                    label = f"tie {prediction.probability:.2f}"
                    fill = (92, 74, 32)
                elif prediction.matches_rule:
                    label = f"{action_name(prediction.predicted_action)} {prediction.probability:.2f}"
                    fill = (24, 78, 60)
                else:
                    label = f"{action_name(prediction.predicted_action)} {prediction.probability:.2f}"
                    fill = (102, 38, 46)
                situation = f"Car {lane_name(prediction.car_lane)}, obstacle {lane_name(prediction.obstacle_lane)}"
                self.blit_fit(situation, (side.x, y + 4), self.font_tiny, TEXT, side.w - 130)
                pygame.draw.rect(self.screen, fill, (side.right - 120, y, 120, 22), border_radius=6)
                self.blit_center(label, pygame.Rect(side.right - 120, y, 120, 22), self.font_tiny, TEXT)
                y += 26
        y += 8
        self.button(
            (side.x, y, side.w, 40),
            "Train this file  (G)",
            self.train_student,
            fill=ACCENT,
            enabled=not self.busy,
        )
        self.button(
            (side.x, y + 48, side.w, 40),
            "Drive the car  (U)",
            lambda: self.set_mode(STUDENT_MODE),
            fill=ORANGE,
            text_color=(28, 20, 16),
            enabled=not self.busy and STUDENT_MODE in self.models,
        )

    def draw_live_loss(self, rect: pygame.Rect, losses: list[float]):
        self.panel(rect)
        self.blit("Training loss", (rect.x + 16, rect.y + 12), self.font_small)
        plot = pygame.Rect(rect.x + 52, rect.y + 48, rect.w - 72, rect.h - 78)
        pygame.draw.rect(self.screen, (18, 20, 28), plot, border_radius=8)
        if len(losses) < 2:
            self.blit_center("The line appears as soon as training starts.", plot, self.font_small, MUTED)
            return
        low = min(losses)
        high = max(losses)
        if high - low < 1e-6:
            high = low + 1.0
        points = []
        last_index = len(losses) - 1
        for index, value in enumerate(losses):
            px = plot.x + plot.w * index / last_index
            py = plot.bottom - 8 - (plot.h - 16) * (value - low) / (high - low)
            points.append((px, py))
        pygame.draw.lines(self.screen, ORANGE, False, points, 3)
        self.blit(f"{high:.2f}", (rect.x + 8, plot.y), self.font_tiny, MUTED)
        self.blit(f"{low:.2f}", (rect.x + 8, plot.bottom - 18), self.font_tiny, MUTED)
        self.blit("epoch", (plot.right - 48, plot.bottom + 6), self.font_tiny, MUTED)

    # --- data -------------------------------------------------------------

    def draw_data(self):
        area = self.content_rect()
        self.blit("Collected demonstrations", (area.x, area.y), self.font_heading)
        records = self.logger.records
        successes = sum(record.success for record in records)
        sessions = len({record.session_id for record in records})
        summary = (
            f"{len(records)} rounds     {successes} successful     "
            f"{len(records) - successes} misses     {sessions} sessions"
        )
        reactions = reaction_summary(records)
        if reactions is not None:
            summary += f"     mean reaction {reactions['mean'] / 1000:.2f}s"
        self.blit_fit(summary, (area.x, area.y + 32), self.font_body, TEXT, area.w)
        missing = missing_successful_cases(records)
        if not records:
            banner, colour = "Play in human mode. Each keypress is added to the CSV immediately.", MUTED
        elif not missing:
            banner, colour = "Every situation has a successful demonstration.", GREEN
        else:
            pending = ", ".join(f"car {lane_name(car)} / obstacle {lane_name(obs)}" for car, obs in missing)
            banner, colour = "Still missing a success for " + pending, AMBER
        self.blit_fit(banner, (area.x, area.y + 60), self.font_small, colour, area.w)
        counts = combination_counts(records)
        top = area.y + 92
        gap = 12
        card_w = (area.w - gap) // 2
        card_h = max(120, min(220, (area.bottom - top - 52 - gap) // 2))
        for index, item in enumerate(counts):
            rect = pygame.Rect(
                area.x + (index % 2) * (card_w + gap),
                top + (index // 2) * (card_h + gap),
                card_w,
                card_h,
            )
            self.draw_case_card(rect, item)
        y = top + 2 * (card_h + gap) + 4
        self.blit_fit(f"File: {self.logger.path}", (area.x, y), self.font_tiny, MUTED, area.w)
        self.draw_wrapped(
            "Open that file in a spreadsheet to see every row. AI rounds are excluded. "
            "Misses stay in the file; the default training filter uses successes only.",
            area.x,
            y + 20,
            self.font_tiny,
            MUTED,
            area.w,
        )

    def draw_case_card(self, rect: pygame.Rect, item: dict):
        success = item["success"]
        total = item["total"]
        if success:
            border = GREEN
        elif total:
            border = AMBER
        else:
            border = (120, 72, 78)
        self.panel(rect)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=14)
        car = lane_name(item["car_lane"]).capitalize()
        obstacle = lane_name(item["obstacle_lane"]).capitalize()
        self.blit(f"Car {car}", (rect.x + 16, rect.y + 14), self.font_heading)
        self.blit(f"Obstacle {obstacle}", (rect.x + 16, rect.y + 42), self.font_body, MUTED)
        rule = action_name(correct_action(item["car_lane"], item["obstacle_lane"])).capitalize()
        self.blit(f"Rule: {rule}", (rect.x + 16, rect.y + 72), self.font_body, ORANGE if rule == "Switch" else BLUE)
        self.blit(f"Played {total}      Successful {success}", (rect.x + 16, rect.bottom - 58), self.font_small)
        if success == 0:
            note = "Not in the default training set yet."
            note_colour = AMBER
        else:
            note = "Usable as a demonstration."
            note_colour = GREEN
        self.blit(note, (rect.x + 16, rect.bottom - 32), self.font_tiny, note_colour)

    # --- train ------------------------------------------------------------

    def draw_train(self):
        area = self.content_rect()
        self.draw_workflow(area)
        body_top = area.y + 86
        body = pygame.Rect(area.x, body_top, area.w, area.bottom - body_top)
        left = pygame.Rect(body.x, body.y, int(body.w * 0.58) - 6, body.h)
        right = pygame.Rect(left.right + 12, body.y, body.right - left.right - 12, body.h)
        self.draw_train_hero(left)
        self.draw_architecture_column(right)

    def draw_workflow(self, area: pygame.Rect):
        labels = [
            "1  CSV rows",
            "2  Two inputs",
            "3  Your action",
            "4  Cross-entropy",
            "5  Adam",
            "6  Plot the loss",
        ]
        gap = 8
        chip_w = (area.w - gap * (len(labels) - 1)) // len(labels)
        for index, label in enumerate(labels):
            rect = pygame.Rect(area.x + index * (chip_w + gap), area.y, chip_w, 36)
            pygame.draw.rect(self.screen, PANEL_2, rect, border_radius=8)
            self.blit_center(label, rect, self.font_tiny, TEXT)
        recipe = (
            f"Inputs are only car lane and obstacle lane. The target is the key you pressed. "
            f"Adam, learning rate {LEARNING_RATE:g}, {EPOCHS} epochs, seed {SEED}, same for every network."
        )
        self.draw_wrapped(recipe, area.x, area.y + 42, self.font_tiny, MUTED, area.w)

    def draw_train_hero(self, rect: pygame.Rect):
        self.button((rect.x, rect.y, 210, 40), "Train models  (T)", self.train, fill=ACCENT, enabled=not self.busy)
        toggle = "Include mistakes" if self.include_mistakes else "Successes only"
        toggle_fill = (120, 84, 32) if self.include_mistakes else (32, 92, 70)
        self.button(
            (rect.x + 222, rect.y, 200, 40),
            toggle,
            self.toggle_mistakes,
            fill=toggle_fill,
            enabled=not self.busy,
        )
        y = rect.y + 50
        y = self.draw_wrapped(self.dataset_note, rect.x, y, self.font_tiny, MUTED, rect.w)
        for note in self._train_warnings():
            y = self.draw_wrapped(note, rect.x, y + 2, self.font_tiny, AMBER, rect.w)
        y += 8
        self.blit("Four situations", (rect.x, y), self.font_small)
        y += 24
        self.draw_hero_table(pygame.Rect(rect.x, y, rect.w, 210))
        y += 218
        if PRIMARY_LINEAR in self.models and PRIMARY_NONLINEAR in self.models and y + 40 <= rect.bottom:
            self.button(
                (rect.x, y, 180, 36),
                "Watch linear drive",
                lambda: self.set_mode(PRIMARY_LINEAR),
                fill=hex_rgb(architecture_by_key(PRIMARY_LINEAR).color),
                text_color=(20, 24, 32),
            )
            self.button(
                (rect.x + 192, y, 180, 36),
                "Watch ReLU drive",
                lambda: self.set_mode(PRIMARY_NONLINEAR),
                fill=hex_rgb(architecture_by_key(PRIMARY_NONLINEAR).color),
                text_color=(28, 20, 16),
            )

    def _train_warnings(self) -> list[str]:
        history = self.histories.get(PRIMARY_LINEAR)
        if history is None:
            return []
        notes = []
        for car, obstacle in history.missing_combinations:
            notes.append(
                f"No training row for car {lane_name(car)}, obstacle {lane_name(obstacle)}. "
                "That situation was not shown to the networks."
            )
        for car, obstacle in history.conflicting_inputs:
            notes.append(
                f"Car {lane_name(car)}, obstacle {lane_name(obstacle)} was demonstrated with both actions. "
                "The networks are imitating a mixture."
            )
        return notes

    def draw_hero_table(self, rect: pygame.Rect):
        columns = [0.34, 0.18, 0.24, 0.24]
        headers = ["Situation", "Rule", "Linear", "ReLU"]
        header_h = 26
        for index, header in enumerate(headers):
            cell = self._column_rect(rect, columns, index, rect.y, header_h)
            self.blit_fit(header, (cell.x + 4, cell.y + 4), self.font_tiny, MUTED, cell.w - 8)
        linear = self.histories.get(PRIMARY_LINEAR)
        relu = self.histories.get(PRIMARY_NONLINEAR)
        row_h = 42
        for row_index, (car, obstacle) in enumerate(CASES):
            y = rect.y + header_h + row_index * (row_h + 4)
            situation = self._column_rect(rect, columns, 0, y, row_h)
            rule = self._column_rect(rect, columns, 1, y, row_h)
            left = self._column_rect(rect, columns, 2, y, row_h)
            right = self._column_rect(rect, columns, 3, y, row_h)
            pygame.draw.rect(self.screen, PANEL_2, situation, border_radius=8)
            label = f"Car {lane_name(car)}, obstacle {lane_name(obstacle)}"
            self.blit_fit(label, (situation.x + 8, situation.y + 12), self.font_tiny, TEXT, situation.w - 12)
            expected = action_name(correct_action(car, obstacle))
            self.blit_center(expected, rule, self.font_small, ORANGE if expected == "switch" else BLUE)
            self._prediction_cell(left, None if linear is None else linear.predictions[row_index])
            self._prediction_cell(right, None if relu is None else relu.predictions[row_index])

    def _column_rect(self, table, fractions, index, y, height) -> pygame.Rect:
        x = table.x
        gap = 4
        usable = table.w - gap * (len(fractions) - 1)
        for earlier, fraction in enumerate(fractions[:index]):
            x += int(usable * fraction) + gap
        return pygame.Rect(x, y, int(usable * fractions[index]), height)

    def _prediction_cell(self, rect: pygame.Rect, prediction):
        if prediction is None:
            pygame.draw.rect(self.screen, PANEL_2, rect, border_radius=8)
            self.blit_center("-", rect, self.font_small, MUTED)
            return
        if abs(prediction.probability - 0.5) < 0.05:
            fill = (92, 74, 32)
        elif prediction.matches_rule:
            fill = (24, 78, 60)
        else:
            fill = (102, 38, 46)
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        if abs(prediction.probability - 0.5) < 0.05:
            label = f"tie  {prediction.probability:.2f}"
        else:
            label = f"{action_name(prediction.predicted_action)}  {prediction.probability:.2f}"
        self.blit_center(label, rect, self.font_tiny, TEXT)

    def draw_architecture_column(self, rect: pygame.Rect):
        linear_n = self.parameter_counts[PRIMARY_LINEAR]
        relu_n = self.parameter_counts[PRIMARY_NONLINEAR]
        if linear_n == relu_n:
            sentence = (
                f"Linear and ReLU have the same {linear_n} weights. "
                "The only difference is the ReLU bend."
            )
        else:
            sentence = f"Linear has {linear_n} weights. ReLU has {relu_n}."
        y = self.draw_wrapped(sentence, rect.x, rect.y, self.font_tiny, TEXT, rect.w)
        y += 6
        gap = 6
        count = len(ARCHITECTURES)
        available = rect.bottom - y
        card_h = min(76, (available - gap * (count - 1)) / count)
        for index, architecture in enumerate(ARCHITECTURES):
            card = pygame.Rect(rect.x, y + index * (card_h + gap), rect.w, card_h)
            self.draw_architecture_card(card, architecture)

    def draw_architecture_card(self, rect: pygame.Rect, architecture):
        self.panel(rect, PANEL_2)
        accent = hex_rgb(architecture.color)
        pygame.draw.rect(self.screen, accent, (rect.x + 8, rect.y + 8, 5, max(8, rect.h - 16)), border_radius=2)
        if self.mode == architecture.key:
            pygame.draw.rect(self.screen, accent, rect, width=2, border_radius=14)
        history = self.histories.get(architecture.key)
        title = f"{architecture.name}    {self.parameter_counts[architecture.key]} weights"
        self.blit_fit(title, (rect.x + 20, rect.y + 6), self.font_small, TEXT, rect.w - 112)
        if history is None:
            detail = f"{architecture.diagram}    not trained"
            detail_colour = MUTED
        else:
            outcome = "tie" if history.uncommitted() else f"{history.cases_correct}/4"
            detail = f"{architecture.diagram}    loss {history.training_loss:.3f}    {outcome}"
            if history.uncommitted():
                detail_colour = AMBER
            else:
                detail_colour = GREEN if history.cases_correct == 4 else RED
        text_w = rect.w - 112
        if rect.h >= 58:
            self.blit_fit(architecture.summary, (rect.x + 20, rect.y + 28), self.font_tiny, MUTED, text_w)
            self.blit_fit(detail, (rect.x + 20, rect.y + 46), self.font_tiny, detail_colour, text_w)
        else:
            self.blit_fit(detail, (rect.x + 20, rect.y + 28), self.font_tiny, detail_colour, text_w)
        if history is not None:
            self.button(
                (rect.right - 78, rect.centery - 13, 66, 26),
                "Drive",
                lambda key=architecture.key: self.set_mode(key),
                fill=accent,
                text_color=(20, 22, 28),
            )

    # --- curves and boundaries -------------------------------------------

    def draw_curves(self):
        area = self.content_rect()
        self.blit("Training curves", (area.x, area.y), self.font_heading)
        caption = self._curve_caption()
        caption_bottom = self.draw_wrapped(caption, area.x, area.y + 30, self.font_tiny, MUTED, area.w)
        if self.figures_note:
            self.blit_fit(self.figures_note, (area.x, caption_bottom), self.font_tiny, MUTED, area.w)
            caption_bottom += 18
        table_h = 28 + len(ARCHITECTURES) * 32
        plot_bottom = area.bottom - table_h - 8
        plot_rect = pygame.Rect(area.x, caption_bottom + 6, area.w, max(80, plot_bottom - caption_bottom - 6))
        self.panel(plot_rect)
        if self._loss_series():
            surface = self._cached_loss_surface()
            fitted = self._fit_surface(surface, plot_rect.inflate(-16, -16))
            self.screen.blit(fitted, fitted.get_rect(center=plot_rect.center))
        else:
            self.blit_center("Train the models to plot the loss.", plot_rect, self.font_body, MUTED)
        self.draw_results_table(pygame.Rect(area.x, area.bottom - table_h, area.w, table_h))

    def _curve_caption(self) -> str:
        history = self.histories.get(PRIMARY_LINEAR)
        if history is None:
            return (
                "Loss is binary cross-entropy: how wrong the probabilities are. "
                "Accuracy is the share of demonstrations matched by the prediction. "
                "Dashed lines are networks with no hidden activation."
            )
        if history.missing_combinations:
            return (
                "A falling loss only means the network matched the rows it was shown. "
                "The situations column still tests all four corners."
            )
        if history.training_accuracy < 0.6:
            return (
                "The dashed lines sit on the loss of guessing 50%. "
                "A straight boundary cannot be right on both switch corners, so cross-entropy "
                "keeps the linear networks unsure. The solid curves fall because a nonlinear "
                "hidden layer can be right on all four corners."
            )
        return "The situations column is the four-corner test. Training accuracy weights rows by how often you played them."

    def _cached_loss_surface(self) -> pygame.Surface:
        key = ("loss", self.plot_version)
        if key not in self.plot_cache:
            self.plot_cache[key] = self._rgb_surface(render_loss_figure(self._loss_series()))
        return self.plot_cache[key]

    def draw_results_table(self, rect: pygame.Rect):
        headers = ["Network", "Layers", "Weights", "Loss", "Train acc", "Situations", ""]
        fractions = [0.16, 0.30, 0.10, 0.12, 0.12, 0.12, 0.08]
        for index, header in enumerate(headers):
            cell = self._column_rect(rect, fractions, index, rect.y, 24)
            self.blit_fit(header, (cell.x + 4, cell.y + 2), self.font_tiny, MUTED, cell.w - 6)
        for row_index, architecture in enumerate(ARCHITECTURES):
            y = rect.y + 26 + row_index * 32
            history = self.histories.get(architecture.key)
            if history is None:
                loss_text, accuracy_text, cases_text = "-", "-", "-"
                cases_colour = TEXT
            elif history.uncommitted():
                loss_text = f"{history.training_loss:.3f}"
                accuracy_text = "tie"
                cases_text = "tie"
                cases_colour = AMBER
            else:
                loss_text = f"{history.training_loss:.3f}"
                accuracy_text = f"{history.training_accuracy:.0%}"
                cases_text = f"{history.cases_correct}/4"
                cases_colour = GREEN if history.cases_correct == 4 else RED
            values = [
                architecture.name,
                architecture.diagram,
                str(self.parameter_counts[architecture.key]),
                loss_text,
                accuracy_text,
                cases_text,
            ]
            for index, value in enumerate(values):
                cell = self._column_rect(rect, fractions, index, y, 28)
                colour = cases_colour if index >= 4 and history is not None else TEXT
                self.blit_fit(value, (cell.x + 4, cell.y + 4), self.font_tiny, colour, cell.w - 6)
            button_cell = self._column_rect(rect, fractions, 6, y, 28)
            self.button(
                (button_cell.x, button_cell.y, button_cell.w - 4, 26),
                "Drive",
                (lambda key=architecture.key: self.set_mode(key)) if history is not None else None,
                fill=hex_rgb(architecture.color),
                text_color=(20, 22, 28),
                enabled=history is not None,
            )

    def draw_boundary(self):
        area = self.content_rect()
        right_key = self.compare_keys[self.compare_index]
        right_name = architecture_by_key(right_key).name
        self.blit("Why the straight line fails", (area.x, area.y), self.font_heading)
        self.blit_fit(
            f"Left: linear.   Right: {right_name}.   Orange points should switch.   Blue points should stay.",
            (area.x, area.y + 30),
            self.font_tiny,
            MUTED,
            area.w - 280,
        )
        self.button((area.right - 210, area.y + 24, 96, 30), "Previous", lambda: self.cycle_compare(-1), fill=PANEL_2)
        self.button((area.right - 106, area.y + 24, 90, 30), "Next", lambda: self.cycle_compare(1), fill=PANEL_2)
        footer_h = 150
        plot_rect = pygame.Rect(area.x, area.y + 62, area.w, area.h - 62 - footer_h)
        self.panel(plot_rect)
        if PRIMARY_LINEAR in self.models and right_key in self.models:
            surface = self._cached_decision_surface(right_key)
            fitted = self._fit_surface(surface, plot_rect.inflate(-12, -12))
            self.screen.blit(fitted, fitted.get_rect(center=plot_rect.center))
        else:
            self.blit_center("Train the models to draw the boundaries.", plot_rect, self.font_body, MUTED)
        y = plot_rect.bottom + 8
        y = self.draw_wrapped(SEPARATION_TEXT, area.x, y, self.font_tiny, TEXT, area.w)
        y = self.draw_wrapped(LINEAR_STACK_EXPLANATION, area.x, y + 2, self.font_tiny, MUTED, area.w)
        if y < area.bottom - 8 and self.figures_note:
            self.blit_fit(self.figures_note, (area.x, min(y, area.bottom - 18)), self.font_tiny, MUTED, area.w)

    def _cached_decision_surface(self, right_key: str) -> pygame.Surface:
        key = ("decision", self.plot_version, right_key)
        if key not in self.plot_cache:
            image = render_decision_figure(
                self.models[PRIMARY_LINEAR],
                self._model_title(PRIMARY_LINEAR),
                self.models[right_key],
                self._model_title(right_key),
            )
            self.plot_cache[key] = self._rgb_surface(image)
        return self.plot_cache[key]


def main():
    App().run()


if __name__ == "__main__":
    main()
