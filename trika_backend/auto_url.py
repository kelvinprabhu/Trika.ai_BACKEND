import os

PROJECT_NAME = "trika_backend"
APPS = ["trikavision", "meditation", "trikabot", "trika_general"]
BASE_URL_FILE = os.path.join(PROJECT_NAME, "urls.py")

def create_urls_py(app_name):
    url_path = os.path.join(app_name, "urls.py")
    if not os.path.exists(url_path):
        with open(url_path, "w") as f:
            f.write(f'''from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='{app_name}-index'),
]
''')
        print(f"[+] Created: {url_path}")
    else:
        print(f"[=] Already exists: {url_path}")

def create_dummy_view(app_name):
    view_path = os.path.join(app_name, "views.py")
    with open(view_path, "a") as f:
        f.write(f'''

def index(request):
    from django.http import JsonResponse
    return JsonResponse({{"message": "{app_name} is working"}})
''')
    print(f"[+] Added dummy view to: {view_path}")

def update_project_urls():
    include_lines = []
    with open(BASE_URL_FILE, "r") as f:
        lines = f.readlines()

    # Remove previously inserted blocks
    lines = [line for line in lines if "AUTO-URL" not in line]

    # Rebuild with new includes
    for app in APPS:
        include_lines.append(f"    path('{app}/', include('{app}.urls')),  # AUTO-URL\n")

    new_urls = []
    in_block = False
    for line in lines:
        new_urls.append(line)
        if "from django.urls" in line and "include" not in line:
            new_urls[-1] = line.strip() + ", include\n"
        if "urlpatterns = [" in line:
            new_urls += include_lines

    with open(BASE_URL_FILE, "w") as f:
        f.writelines(new_urls)

    print(f"[+] Updated: {BASE_URL_FILE}")

# Run everything
for app in APPS:
    create_urls_py(app)
    create_dummy_view(app)

update_project_urls()
