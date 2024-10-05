from src.const import CONST_TAG_ID_SYSTEM, CONST_TAG_ID_TODO
from src.models import db_event_base, Event, Tag, EventTag, Meta


def init_database():
    db_event_base.create_tables([
        Event,
        Tag,
        EventTag,
        Meta,
    ], safe=True)
    # 该记录将用于维持系统功能
    tg1 = Tag.create(id=CONST_TAG_ID_SYSTEM, name='$SYSTEM', desc='DO_NOT_DELETE')
    tg2 = Tag.create(id=CONST_TAG_ID_TODO, name='$TODO', desc='DO_NOT_DELETE')

    evt = Event(content='Initialization completed')
    evt.save()
    EventTag.create(event=evt, tag=tg1)


if __name__ == '__main__':
    init_database()
