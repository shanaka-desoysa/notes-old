# README

# WELCOME TO SHANAKA'S NOTES!

## Metadata Example
---
title: "PyTorch Tensor Basics"
author: "Shanaka DeSoysa"
date: 2020-05-14T00:00:00-07:00
description: "Converting NumPy arrays to PyTorch tensors and creating tensors from scratch"
type: technical_note
draft: false
---

## Todo (SELF)
* <s>Fix About page</s>
* <s>Change color</s>
* ~Add own categories and content~
* Automatically add meta data to technical notes
* Fix make.ipynb bug 
* Concert make.ipynb to make.py
* Consider some [Hugo Maitenance](https://discourse.gohugo.io/t/advice-needed-regarding-hugo-dev-environment-on-mac/10156/12)

## Learnings (SELF)
* Must add `_index.md` in every folder where content (i.e. example.md) exists
* Must change `.Page` iteration according to [ADVICE](https://discourse.gohugo.io/t/subsubfolders-in-chris-albon-theme/24472)
* Can add images to markdown files with `![png](images/page.jpg)`
* Can change publish directory with `publishDir="../shanaka-desoysa.github.io"`
* Stay Oganized!

Theme code update: index.html
```html
<ul>
    {{ $section := site.GetPage "machine_learning/basics" }}
    {{ with $section }}
        {{ range .Pages }}
        <li><a href="{{.Permalink}}">{{.Title}}</a></li>
        {{ end }}
    {{ end }}
</ul>
```

## Run Jupyter

Minimal Notebook

```bash
docker run --rm -p 10000:8888 -e JUPYTER_ENABLE_LAB=yes -v "$PWD":/home/jovyan/work jupyter/minimal-notebook
```

SciPy Notebook

```bash
docker run --rm -p 10000:8888 -e JUPYTER_ENABLE_LAB=yes -e TZ=America/Chicago -v "$PWD":/home/jovyan/work jupyter/scipy-notebook
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
