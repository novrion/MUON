import modal

app = modal.App("cifar-training")
volume = modal.Volume.from_name("cifar-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "tqdm")
    .add_local_file("src/model.py", "/root/model.py")
    .add_local_file("src/data.py", "/root/data.py")
    .add_local_file("src/muon.py", "/root/muon.py")
    .add_local_file("src/train.py", "/root/train.py")
)


@app.function(gpu="H100", image=image, timeout=3600, volumes={"/outputs": volume})
def train_remote(method: str):
    import sys
    sys.path.insert(0, "/root")
    from train import train, save_model, save_data
    model, data = train(method)
    save_model(model, f"/outputs/{method}_model.pth")
    save_data(data, f"/outputs/{method}_loss.csv")
    volume.commit()


@app.local_entrypoint()
def main(method: str = "muon"):
    train_remote.remote(method)
