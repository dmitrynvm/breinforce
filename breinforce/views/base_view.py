class BaseView:
    '''Base class for renderer. Any renderer must subclass this renderer
    and implement the function render
    '''

    def __init__(self) -> None:
        pass

    def render(self) -> None:
        '''Render the table based on the table configuration
        '''
        raise NotImplementedError
