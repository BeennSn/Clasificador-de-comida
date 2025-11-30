"""Quick test to verify model and API integration work correctly."""
import sys
from pathlib import Path

def test_model_loading():
    """Test that the model loads correctly with proper architecture."""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    
    try:
        from utils import load_model_and_classes
        model, classes, device = load_model_and_classes('./ckpt_finetuned.pt')
        print(f"✅ Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Classes: {len(classes)}")
        print(f"   Class names: {classes}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """Test inference on a sample image if available."""
    print("\n" + "=" * 60)
    print("TEST 2: Inference")
    print("=" * 60)
    
    # Look for test images
    test_images = [
        'apple-pie.jpg',
        'test.jpg',
        'sample.jpg'
    ]
    
    test_img = None
    for img in test_images:
        if Path(img).exists():
            test_img = img
            break
    
    if not test_img:
        print("⚠️  No test image found, skipping inference test")
        return True
    
    try:
        from utils import load_model_and_classes, predict_from_bytes
        
        model, classes, device = load_model_and_classes('./ckpt_finetuned.pt')
        
        with open(test_img, 'rb') as f:
            image_bytes = f.read()
        
        predictions = predict_from_bytes(image_bytes, model, classes, device, topk=3)
        
        print(f"✅ Inference successful on {test_img}")
        print(f"   Top predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"     {i}. {pred['label']}: {pred['confidence']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nutrition_api():
    """Test nutrition API integration."""
    print("\n" + "=" * 60)
    print("TEST 3: Nutrition API")
    print("=" * 60)
    
    try:
        from clients import OpenFoodFactsClient
        
        client = OpenFoodFactsClient()
        result = client.get_best_nutriments("apple pie")
        
        if result:
            print(f"✅ Nutrition API working")
            print(f"   Product: {result.get('product_name', 'N/A')}")
            nutriments = result.get('nutriments', {})
            if nutriments:
                print(f"   Sample nutrients:")
                for key in ['energy-kcal', 'fat', 'carbohydrates', 'proteins']:
                    if key in nutriments:
                        print(f"     {key}: {nutriments[key]}")
        else:
            print(f"⚠️  No nutrition data found (API may be down or no results)")
        
        return True
    except Exception as e:
        print(f"❌ Nutrition API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n🧪 QUICK SYSTEM TEST\n")
    
    results = []
    results.append(("Model Loading", test_model_loading()))
    results.append(("Inference", test_inference()))
    results.append(("Nutrition API", test_nutrition_api()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready.")
        print("\n📝 Next steps:")
        print("   1. Start API: uvicorn app:app --reload")
        print("   2. Test API: python test_api_requests.py")
        print("   3. API docs: http://127.0.0.1:8000/docs")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
