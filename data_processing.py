import urllib.request


def parse_html(html_code):
    courses = {}

    lines = []
    line = ""
    for char in html_code:
        if char == "\n":
            lines.append(line)
            line = ""
        else:
            line = line + char
    if line != "":
        lines.append(line)

    for line in lines:
        if "<div class=" + '"title-subjectcode">' in line:
            if "</div>" in line:
                first_part_end = line.find(">")
                second_part_start = line.find("<", first_part_end + 1)
                course = line[first_part_end + 1: second_part_start]
                if course not in courses:
                    courses[course] = []

    for line in lines:
        if '<p class="course-descr">' in line and "<a id" in line:
            description_start = line.find(">")
            description_end = line.find("<a id")
            description = line[description_start: description_end][1:]
            if 'aria-label="view ' in line:
                course_number_start = line.find(" ", line.find('aria-label="view '))

                course_number_middle = line.find(" ", course_number_start + 1)
                course_number_end = line.find(" ", course_number_middle + 1)
                course = line[course_number_start: course_number_end].lstrip()
                if course in courses:
                    courses[course].append(description)

    return courses


def split_sentence(sentence):
    words = set()
    acc = ""
    for i in sentence:
        if i == " ":
            if acc != "":
                words.add(acc)
                acc = ""
        else:
            acc = acc + i
    if acc != "":
        words.add(acc)
    return words


def process_data(subjects):
    result = []
    for subject in subjects:
        url_link = f'https://classes.cornell.edu/browse/roster/SP24/subject/{subject}'

        try:
            with urllib.request.urlopen(url_link) as f:
                html_code = f.read().decode('utf-8')
                courses = parse_html(html_code)
                result.append(courses)
        except urllib.error.URLError as e:
            print(e.reason)

    return tuple(result)
