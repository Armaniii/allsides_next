import logging
from django.core.management.base import BaseCommand
from api.research.models import ResearchReport
from api.research.utils import repair_missing_thread_state, get_thread_state

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Repair missing thread states for existing research reports'

    def add_arguments(self, parser):
        parser.add_argument(
            '--report-id',
            type=int,
            dest='report_id',
            help='Repair thread state for a specific report ID',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            dest='dry_run',
            help='Only check thread states without repairing them',
        )

    def handle(self, *args, **options):
        report_id = options.get('report_id')
        dry_run = options.get('dry_run', False)
        
        if report_id:
            # Repair a specific report
            try:
                report = ResearchReport.objects.get(id=report_id)
                self.repair_report(report, dry_run)
            except ResearchReport.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'No report found with ID {report_id}'))
        else:
            # Repair all reports
            reports = ResearchReport.objects.all().order_by('-created_at')
            self.stdout.write(f'Found {reports.count()} reports to check')
            
            missing_count = 0
            repaired_count = 0
            
            for report in reports:
                thread_exists = bool(get_thread_state(report.thread_id))
                if not thread_exists:
                    missing_count += 1
                    if not dry_run:
                        success = self.repair_report(report, dry_run)
                        if success:
                            repaired_count += 1
                else:
                    self.stdout.write(f'Report {report.id} has valid thread state')
            
            if dry_run:
                self.stdout.write(self.style.SUCCESS(f'Found {missing_count} reports with missing thread states (dry run, no repairs made)'))
            else:
                self.stdout.write(self.style.SUCCESS(f'Repaired {repaired_count}/{missing_count} reports with missing thread states'))
    
    def repair_report(self, report, dry_run=False):
        """Repair a specific report's thread state."""
        self.stdout.write(f'Checking report {report.id} (thread_id: {report.thread_id})')
        
        thread_exists = bool(get_thread_state(report.thread_id))
        if thread_exists:
            self.stdout.write(self.style.SUCCESS(f'Report {report.id} already has a valid thread state'))
            return True
        
        if dry_run:
            self.stdout.write(self.style.WARNING(f'Report {report.id} is missing thread state (dry run, not repairing)'))
            return False
        
        # Repair the thread state
        success = repair_missing_thread_state(report)
        
        if success:
            self.stdout.write(self.style.SUCCESS(f'Successfully repaired thread state for report {report.id}'))
        else:
            self.stdout.write(self.style.ERROR(f'Failed to repair thread state for report {report.id}'))
        
        return success 