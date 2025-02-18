from pyradise.fileio import (DatasetDicomCrawler, SubjectLoader)


def main_iterative_crawling(dataset_path: str) -> None:
    # Create the crawler (using the modality configuration file)
    crawler = DatasetDicomCrawler(dataset_path)

    # Use the crawler iteratively (more memory efficient)
    for series_info in crawler:
        subject = SubjectLoader().load(series_info)
        # Do something with the subject
        print(subject.get_name())


def main_crawling_using_execute_fn(dataset_path: str) -> None:
    # Create the crawler (using the modality configuration file)
    crawler = DatasetDicomCrawler(dataset_path)

    # Use the crawler with the execute function
    # (all series info entries are loaded in one step)
    series_infos = crawler.execute()

    # Iterate over the series infos
    for series_info in series_infos:
        subject = SubjectLoader().load(series_info)
        # Do something with the subject
        print(subject.get_name())

if __name__ == "__main__":
    path_to_rtss = r"C:\Users\r.joshi\Downloads\01_11_2024\7139000004\COMBI\L"
    output_path =r"./"

    print('iterative crawling')
    main_iterative_crawling(path_to_rtss)

    print('crawling using execute')
    main_crawling_using_execute_fn(path_to_rtss)