"""Sample course data for testing"""

from models import Course, Lesson, CourseChunk

# Sample course data based on actual course format
SAMPLE_COURSE = Course(
    title="Building Towards Computer Use with Anthropic",
    course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
    instructor="Colt Steele",
    lessons=[
        Lesson(
            lesson_number=0,
            title="Introduction",
            lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
        ),
        Lesson(
            lesson_number=1,
            title="Anthropic API Basics",
            lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7k1z/basics"
        ),
        Lesson(
            lesson_number=2,
            title="Multi-modal Requests",
            lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/c8k2z/multimodal"
        )
    ]
)

SAMPLE_COURSE_2 = Course(
    title="Introduction to Machine Learning",
    course_link="https://www.deeplearning.ai/short-courses/intro-to-ml/",
    instructor="Andrew Ng",
    lessons=[
        Lesson(
            lesson_number=1,
            title="What is Machine Learning",
            lesson_link="https://learn.deeplearning.ai/courses/intro-to-ml/lesson/1/what-is-ml"
        ),
        Lesson(
            lesson_number=2,
            title="Supervised Learning",
            lesson_link="https://learn.deeplearning.ai/courses/intro-to-ml/lesson/2/supervised-learning"
        )
    ]
)

# Sample course chunks for testing vector search
SAMPLE_COURSE_CHUNKS = [
    CourseChunk(
        content="Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic and taught by Colt Steele, whose Anthropic's Head of Curriculum. Welcome, Colt. Thanks, Andrew. I'm delighted to have the opportunity to share this course with all of you. Anthropic made a recent breakthrough and released a model that could use a computer.",
        course_title="Building Towards Computer Use with Anthropic",
        lesson_number=0,
        chunk_index=0
    ),
    CourseChunk(
        content="That is, it can look at the screen, a computer usually running in a virtual machine, take a screenshot and generate mouse clicks or keystrokes in sequence to execute some tasks, such as search the web using a browser and download an image, and so on.",
        course_title="Building Towards Computer Use with Anthropic",
        lesson_number=0,
        chunk_index=1
    ),
    CourseChunk(
        content="This computer use capability is built by using many features of large language models in combination, including their ability to process an image, such as to understand what's happening in a screenshot, or to use tools that generate mouse clicks and keystrokes.",
        course_title="Building Towards Computer Use with Anthropic", 
        lesson_number=0,
        chunk_index=2
    ),
    CourseChunk(
        content="In this lesson, you'll learn the basics of making API requests to Anthropic's Claude models. We'll cover authentication, request formatting, and handling responses.",
        course_title="Building Towards Computer Use with Anthropic",
        lesson_number=1,
        chunk_index=3
    ),
    CourseChunk(
        content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
        course_title="Introduction to Machine Learning",
        lesson_number=1,
        chunk_index=4
    )
]

# Course metadata for testing course resolution
SAMPLE_COURSE_METADATA = [
    {
        "title": "Building Towards Computer Use with Anthropic",
        "instructor": "Colt Steele", 
        "course_link": "https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"},
            {"lesson_number": 1, "lesson_title": "Anthropic API Basics", "lesson_link": "https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7k1z/basics"},
            {"lesson_number": 2, "lesson_title": "Multi-modal Requests", "lesson_link": "https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/c8k2z/multimodal"}
        ],
        "lesson_count": 3
    },
    {
        "title": "Introduction to Machine Learning",
        "instructor": "Andrew Ng",
        "course_link": "https://www.deeplearning.ai/short-courses/intro-to-ml/", 
        "lessons": [
            {"lesson_number": 1, "lesson_title": "What is Machine Learning", "lesson_link": "https://learn.deeplearning.ai/courses/intro-to-ml/lesson/1/what-is-ml"},
            {"lesson_number": 2, "lesson_title": "Supervised Learning", "lesson_link": "https://learn.deeplearning.ai/courses/intro-to-ml/lesson/2/supervised-learning"}
        ],
        "lesson_count": 2
    }
]

# Sample search queries for testing
SAMPLE_QUERIES = {
    "content_search": [
        "computer use capability",
        "API requests", 
        "machine learning definition",
        "screenshot analysis"
    ],
    "course_outline": [
        "Building Towards Computer Use",
        "Computer Use", 
        "Introduction to Machine Learning",
        "ML"
    ]
}