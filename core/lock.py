import os
import time
import uuid
import glob

###############################################################################


class Lock():

    def __init__(self, file_):
        # We create an id to lock the file
        self.__lock_id = str(uuid.uuid4())
        self.__lock_id = self.__lock_id.replace("-", "")

        # We save the path associated with the data
        self.__path_file = str(file_)
        self._lock_file = self.__path_file+".lock."+self.__lock_id
        # We initialize the flag to know if we got the lock
        self.__got_lock = False

    # ----------------------------------------------------------------------- #
    # Lock

    def do(self, function, *args, **kwargs):

        # If we got the lock, we run the function and return the result
        if(self.__got_lock):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                self._release_lock()
                raise e

        # We initialize two flag to know if the saving was a success
        ok_release = False

        # If the saving was not a success,
        while(not(ok_release)):

            # We get the lock
            self._get_lock()
            # We run the function and keep the result
            try:
                result = function(*args, **kwargs)
            except Exception as e:
                self._release_lock()
                raise e
            # We try to release the lock and save the result
            # (and start over if there is a problem)
            ok_release = self._release_lock()

        # We return the result
        return result

    def _release_lock(self):

        # If the saving was a success, we rename the lock file as the
        # original file
        ok_release = self._rename_lock()

        # If one of the two operations fail, we remove the lock
        if(not(ok_release)):
            self._clean_lock()

        # We return the state of the two operations
        return ok_release

    def _get_lock(self):

        # If the file does not exist, we create it
        if(not(os.path.exists(self.__path_file))
           and len(glob.glob(self.__path_file+".lock.*")) == 0):
            tmp_f = open(self.__path_file, 'a+')
            tmp_f.close()

        # Until we get the lock
        while not(self.__got_lock):
            try:
                # We try to get it by renaming the original file by the lock
                # file ".lock.id"
                os.replace(
                    self.__path_file,
                    self._lock_file)
                self.__got_lock = True
            except OSError:
                # If we can't rename the original file, we wait
                time.sleep(0.1)
                # If we do not found the lock file, we keep lock flag to False
                self.__got_lock = False

    def _clean_lock(self):
        # If there is the current lock file and the orginal file,
        if(os.path.exists(self._lock_file)
           and os.path.exists(self.__path_file)):
            # We remove the current lock file
            os.remove(self._lock_file)

    def _rename_lock(self):
        # If we got the lock
        if(self.__got_lock):

            # If there is the current lock file and the orginal file,
            if(os.path.exists(self._lock_file)
               and os.path.exists(self.__path_file)):
                # The lock has a problem
                ok = False
            else:
                # We try to rename the lock file as the original file
                try:
                    os.replace(
                        self._lock_file,
                        self.__path_file)
                    ok = True
                except OSError:
                    ok = False

            # We set the lock flag accordingly
            self.__got_lock = False

        else:
            # We have not the lock, everything is fine!
            ok = True

        # We return if the renaming was a success
        return ok
