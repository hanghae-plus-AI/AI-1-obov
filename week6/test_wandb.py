import wandb

# 고유한 run_id 설정 (고정된 ID로 계속 같은 run을 업데이트)
run_id = "unique_run_id_example"

if __name__ == "__main__":
    # 동일한 run을 업데이트하기 위해 resume="allow"와 run_id 설정
    wandb.init(project="example-project", id=run_id, resume="allow")

    wandb.run.name = "example-run"

    columns = ["Text", "Predicted Sentiment", "True Sentiment"]
    data = [["I love my phone", "1", "1"], ["My phone is not working", "0", "-1"]]
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"predictions": table})
