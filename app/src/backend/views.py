from django.http import JsonResponse, HttpResponse
from text_analizer import analize_text

def process_review(request):
    if request.method != 'GET':
        return HttpResponse('Only GET is allowed', status=405)
    try:
        rating, sentiment = analize_text(request.GET['text'], request.GET['model'])
    except ValueError as e:
        return JsonResponse({'Error': str(e)}, status=400)
    return JsonResponse({'rating': rating, 'sentiment': 'positive' if sentiment else 'negative'})
