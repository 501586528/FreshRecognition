from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms, models
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 类别名称（根据训练集）
class_names = ['Fresh', 'Less-fresh', 'Spoiled']

# 加载模型结构
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('shrimp_3class_model_20250528_133648.pth', map_location='cpu'))
model.eval()

# 预处理方法（和训练时 test 一致）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': '未接收到图片数据'
            }), 400
        
        print(f"接收到base64数据，长度: {len(data['image'])}")
        
        base64_data = data['image']
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        print(f"图片尺寸: {img.size}")
        
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted = torch.max(output, 1)
            result = int(predicted.item())
            confidence = float(probabilities[result].item())
        
        # 中文映射
        chinese_names = {
            'Fresh': '新鲜',
            'Less fresh': '较新鲜', 
            'Spoiled': '变质'
        }
        
        class_name = class_names[result]
        chinese_name = chinese_names.get(class_name, class_name)
        
        print(f"识别结果: {class_name} ({chinese_name}), 置信度: {confidence:.2f}")
        
        return jsonify({
            'success': True,
            'class_id': result,
            'class_name': class_name,
            'chinese_name': chinese_name,
            'confidence': confidence,
            'result': f"{chinese_name} (置信度: {confidence:.1%})",
            'message': '识别成功'
        })
        
    except Exception as e:
        print(f"识别过程中发生错误: {e}")
        return jsonify({
            'success': False,
            'error': f'识别失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)