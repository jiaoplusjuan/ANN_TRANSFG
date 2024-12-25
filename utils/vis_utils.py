import matplotlib.pyplot as plt

def show(train_losses, val_losses, train_accuracies, val_accuracies,args):
    # 绘制训练和验证损失的图表
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('pic/' + args.name + 'train_val_loss' + '.jpg')  # 保存图表为图片文件
    plt.close()  # 关闭当前图表

    # 绘制训练和验证准确率的图表
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('pic/' + args.name + 'train_val_accuracy' + '.jpg')  # 保存图表为图片文件
    plt.close()  # 关闭当前图表
