import pytest
from breinforce import views


def test_base():
    view = views.BaseView(0, 0, 0)
    with pytest.raises(NotImplementedError):
        view.render({})
