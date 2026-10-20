from pathlib import Path

from data import prepare_step_data
from models import create_step_model
from optimizers import create_optimizer
from reporting import save_step_trace
from training import update_step


def main():
    x, y = prepare_step_data()
    model = create_step_model()
    optimizer = create_optimizer(model, name="SGD", lr=0.1)

    _, _, records = update_step(model, x, y, optimizer)

    output_dir = Path(__file__).resolve().parent / "results"
    output_path = save_step_trace(records, output_dir)

    for row in records:
        print(row)

    print("记录已保存到：", output_path)


if __name__ == "__main__":
    main()
