from django.core.management.base import BaseCommand
from django.core.management import call_command

class Command(BaseCommand):
    help = 'Import disease data from multiple URLs'

    def handle(self, *args, **kwargs):
        urls = [
            'https://vnvc.vn/cac-benh-truyen-nhiem-thuong-gap/',
            'https://www.vinmec.com/vie/bai-viet/10-can-benh-nguy-hiem-nhat-trong-xa-hoi-hien-nay-vi',
            'https://www.vinmec.com/vie/bai-viet/9-benh-pho-bien-do-virus-hoac-vi-khuan-gay-nen-vi',
            'https://nhathuoclongchau.com.vn/bai-viet/cac-benh-truyen-nhiem-thuong-gap.html',
            'https://www.vinmec.com/vie/bai-viet/danh-muc-cac-benh-truyen-nhiem-thuong-gap-vi',
            'https://nhathuoclongchau.com.vn/bai-viet/cac-benh-pho-bien-o-viet-nam.html',
            'https://medda.vn/cac-benh-thuong-gap/danh-sach-cac-benh-thuong-gap-o-viet-nam',
            'https://www.minhanhhospital.com.vn/vi/news/tin-tuc-su-kien-minh-anh/7-benh-thuong-gap-mua-thu-va-dong-911.html'
        ]
        
        for url in urls:
            self.stdout.write(f'Importing from {url}...')
            try:
                call_command('import_from_url', url)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error importing from {url}: {e}'))
        
        self.stdout.write(self.style.SUCCESS(f'Completed importing from {len(urls)} URLs'))
