from transformers import TrainerCallback


class BestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_checkpoint = None
        self.best_metric = None
        self.step_to_epoch_map = {}  # step을 epoch로 매핑하는 딕셔너리

    def on_step_end(self, args, state, control, **kwargs):
        # 매 스텝마다 step -> epoch 매핑 생성
        global_step = state.global_step
        current_epoch = state.epoch

        # 매핑을 딕셔너리에 저장
        self.step_to_epoch_map[global_step] = current_epoch

    def on_train_end(self, args, state, control, **kwargs):
        # 가장 좋은 모델의 체크포인트와 메트릭 정보를 가져옴
        self.best_checkpoint = state.best_model_checkpoint
        self.best_metric = state.best_metric

        # 메트릭 종류 확인 (eval_loss를 기본으로 사용)
        metric_name = (
            args.metric_for_best_model if args.metric_for_best_model else "eval_loss"
        )

        if self.best_checkpoint:
            global_step = int(self.best_checkpoint.split("-")[-1])

            # step -> epoch 매핑에서 해당 step의 epoch 가져오기
            best_epoch = self.step_to_epoch_map.get(global_step, "Unknown epoch")

            print(f"Best model was saved at: {self.best_checkpoint}")
            print(f"Best epoch: {best_epoch}")
            print(f"Best {metric_name}: {self.best_metric}")
        else:
            print("No best model found.")
