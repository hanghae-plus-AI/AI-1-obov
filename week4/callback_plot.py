from transformers import TrainerCallback
import matplotlib.pyplot as plt


class PlotCallback(TrainerCallback):
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        self.during_training = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        # train_loss를 로그에서 가져와 저장
        if "loss" in logs:
            self.train_loss.append(logs["loss"])

    def on_train_begin(self, args, state, control, **kwargs):
        # 학습이 시작될 때 상태를 학습 중으로 설정
        self.during_training = True

    def on_train_end(self, args, state, control, **kwargs):
        # 학습이 끝났을 때 플래그를 학습 중이 아닌 상태로 설정
        self.during_training = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 평가만 수행될 때는 그래프를 그리지 않음
        if not self.during_training:
            return  # 평가 중일 때는 플롯을 그리지 않음

        # state.epoch가 None이 아닌 경우에만 추가
        if state.epoch is not None and state.epoch not in self.epochs:
            self.epochs.append(state.epoch)

        if metrics:
            if "eval_loss" in metrics:
                self.eval_loss.append(metrics["eval_loss"])
            if "eval_accuracy" in metrics:
                self.eval_accuracy.append(metrics["eval_accuracy"])

        # 매 validation이 끝날 때만 그래프를 그림
        self.plot(args.num_train_epochs)

    def plot(self, total_epochs):
        plt.figure(figsize=(10, 6))

        # Train Loss Plot (Epoch에 따라 찍힌 train_loss 값들)
        if len(self.train_loss) > 0:
            plt.subplot(3, 1, 1)
            plt.plot(
                self.epochs[: len(self.train_loss)],
                self.train_loss,
                marker="o",
                label="Train Loss",
                color="blue",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train Loss per Epoch")
            plt.grid(True)
            plt.xlim(0, total_epochs + 1)  # x축 범위 설정
            plt.xticks(
                range(0, total_epochs + 1)
            )  # x축을 0부터 total_epochs까지 정수로만 설정

        # Eval Loss Plot
        if len(self.eval_loss) > 0:
            plt.subplot(3, 1, 2)
            plt.plot(
                self.epochs[: len(self.eval_loss)],
                self.eval_loss,
                marker="o",
                label="Eval Loss",
                color="red",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Eval Loss per Epoch")
            plt.grid(True)
            plt.xlim(0, total_epochs + 1)  # x축 범위 설정
            plt.xticks(
                range(0, total_epochs + 1)
            )  # x축을 0부터 total_epochs까지 정수로만 설정

        # Eval Accuracy Plot
        if len(self.eval_accuracy) > 0:
            plt.subplot(3, 1, 3)
            plt.plot(
                self.epochs[: len(self.eval_accuracy)],
                self.eval_accuracy,
                marker="o",
                label="Eval Accuracy",
                color="green",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Eval Accuracy per Epoch")
            plt.grid(True)
            plt.xlim(0, total_epochs + 1)  # x축 범위 설정
            plt.xticks(
                range(0, total_epochs + 1)
            )  # x축을 0부터 total_epochs까지 정수로만 설정

        plt.tight_layout()
        plt.show()
